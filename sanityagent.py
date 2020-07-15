import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import A_hparameters as hp
from datetime import datetime
from os import path, makedirs
import random
import cv2
import numpy as np

#leave memory space for opencl
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

keras.backend.clear_session()
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

class Player():
    def __init__(self, observation_space, action_space, m_dir=None,
                 log_name=None, start_step=0, start_round=0, buf_full=False,
                 load_buffer=False, buf_count=0):
        """
        model : The actual training model
        t_model : Fixed target model
        """
        print('Model directory : {}'.format(m_dir))
        print('Log name : {}'.format(log_name))
        print('Starting from step {}'.format(start_step))
        print('Starting from round {}'.format(start_round))
        print('Buffer full? {}'.format(buf_full))
        print('Load buffer? {}'.format(load_buffer))
        print('Current buffer count : {}'.format(buf_count))
        self.action_n = action_space.n
        #Inputs
        if m_dir is None :
            left_input = keras.Input(observation_space['Left'].shape,
                                    name='Left')
            right_input = keras.Input(observation_space['Right'].shape,
                                    name='Right')
            # Spare eye model for later use
            left_input_shape = observation_space['Left'].shape
            right_input_shape = observation_space['Right'].shape
            left_eye_model = self.eye_model(left_input_shape,'Left')
            right_eye_model = self.eye_model(right_input_shape,'Right')
            # Get outputs of the model
            left_encoded = left_eye_model(left_input)
            right_encoded = right_eye_model(right_input)
            # Concatenate both eye's inputs
            concat = layers.Concatenate()([left_encoded,right_encoded])
            # concat = layers.Concatenate()([left_input,right_input])
            outputs = self.brain_layers(concat)
            # x = keras.layers.Flatten()(concat)
            # outputs = keras.layers.Dense(self.action_n)(x)
            # Build models
            self.model = keras.Model(inputs=[left_input, right_input],
                                    outputs=outputs)
            self.model.compile(optimizer='Adam', loss='mse', 
                                metrics=['mse'])
        else:
            self.model = keras.models.load_model(m_dir)
        self.t_model = keras.models.clone_model(self.model)
        self.t_model.set_weights(self.model.get_weights())
        self.model.summary()

        # Buffers
        if load_buffer:
            print('loading buffers...')
            buffers = np.load(path.join(m_dir,'buffer.npz'))
            self.right_buffer = buffers['Right']
            self.left_buffer = buffers['Left']
            self.target_buffer = buffers['Target']
            buffers.close()
            print('loaded')
        else :
            self.right_buffer = np.zeros(np.concatenate(([hp.Buffer_size],
                                observation_space['Right'].shape)))
            self.left_buffer = np.zeros(np.concatenate(([hp.Buffer_size],
                                observation_space['Left'].shape)))
            self.target_buffer = np.zeros((hp.Buffer_size,self.action_n))

        # File writer for tensorboard
        if log_name is None :
            self.log_name = datetime.now().strftime('%m_%d_%H_%M_%S')
        else:
            self.log_name = log_name
        self.file_writer = tf.summary.create_file_writer(path.join('log',
                                                         self.log_name))
        self.file_writer.set_as_default()
        print('Writing logs at '+ self.log_name)

        # Scalars
        self.start_training = False
        self.buffer_full = buf_full
        self.total_steps = start_step
        self.current_steps = 1
        self.buffer_count = buf_count
        self.score = 0
        self.rounds = start_round
        self.cumreward = 0
        
        # Savefile folder directory
        if m_dir is None :
            self.save_dir = path.join('savefiles',
                            datetime.now().strftime('%m_%d_%H_%M_%S'))
            self.save_count = 0
        else:
            self.save_dir, self.save_count = path.split(m_dir)
            self.save_count = int(self.save_count)

    def eye_model(self, input_shape, left_or_right):
        """
        Return an eye model
        """
        inputs = layers.Input(input_shape)
        x = layers.Reshape((inputs.shape[1],
                            inputs.shape[2]*inputs.shape[3]))(inputs)
        x = layers.Conv1D(64, 7, strides=2, activation='relu')(x)
        x = layers.Conv1D(32, 5, strides=2, activation='relu')(x)
        x = layers.Conv1D(16, 3, strides=2, activation='relu')(x)
        # outputs = layers.BatchNormalization()(x)
        outputs = x
        return keras.Model(inputs=inputs, outputs=outputs, 
                    name=left_or_right+'_eye')

    def brain_layers(self, x):
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self.action_n)(x)
        outputs = layers.Activation('linear',dtype='float32')(x)
        return outputs

    @property
    def epsilon(self):
        if self.total_steps > hp.epsilon_nstep :
            return hp.epsilon_min
        else:
            return hp.epsilon-(hp.epsilon-hp.epsilon_min)*\
                (self.total_steps/hp.epsilon_nstep)

    def pre_processing(self, observation):
        """
        Preprocess input data
        """
        if len(observation['Right'].shape) < 4:
            observation['Right'] = observation['Right'][np.newaxis,:,:,:].\
                                    astype(np.float32) / 255
            observation['Left'] = observation['Left'][np.newaxis,:,:,:].\
                                    astype(np.float32) / 255
        return observation

    def choose_action(self, q):
        """
        Policy part; uses e-greedy
        """
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_n)
        else :
            m = max(q[0])
            indices = [i for i, x in enumerate(q[0]) if x==m]
            return random.choice(indices)

    def act(self, before_state):
        before_state = self.pre_processing(before_state)
        self.right_buffer[self.buffer_count%hp.Buffer_size] = \
            before_state['Right']
        self.left_buffer[self.buffer_count%hp.Buffer_size] = \
            before_state['Left']
        self.bef_state = before_state
        self.q = self.model(self.bef_state, training=False).numpy()
        self.action = self.choose_action(self.q)
        return self.action

    def step(self, after_state, reward, done, info):
        after_state = self.pre_processing(after_state)
        # Record here, so that it won't record when evaluating
        tf.summary.scalar('maxQ', tf.math.reduce_max(self.q), self.total_steps)
        if info['ate_apple']:
            self.score += 1
        self.cumreward += reward

        if done:
            tf.summary.scalar('Score', self.score, self.rounds)
            tf.summary.scalar('Reward', self.cumreward, self.rounds)
            print('{0} round({1} steps) || Score: {2} | Reward: {3:.1f}'.format(
                self.rounds, self.current_steps, self.score, self.cumreward
            ))
            self.score = 0
            self.current_steps = 0
            self.cumreward = 0
            self.rounds += 1

            # Q-learning Thing
            self.q[0, self.action] = reward
        else:
            self.q[0, self.action] = reward + hp.Q_discount*np.max(
                                    self.t_model(after_state, training=False))
        self.target_buffer[self.buffer_count%hp.Buffer_size] = self.q[0]
        if not self.start_training :
            if self.buffer_count > hp.Learn_start or self.buffer_full:
                self.start_training = True
            else:
                self.buffer_count += 1
                if not self.buffer_count % 100 :
                    print('filling buffer {0}/{1}'.format(
                        self.buffer_count, hp.Learn_start))
        # To check at least once if buffer count is larger than learn start,
        # DO NOT use else
        if self.start_training:
            if not self.buffer_full:
                batch_indices = random.sample(range(self.buffer_count),
                                              hp.Batch_size)
                if self.buffer_count >= hp.Buffer_size-1:
                    self.buffer_full = True
            else:
                batch_indices = random.sample(range(hp.Buffer_size),
                                              hp.Batch_size)
            self.buffer_count += 1
            self.buffer_count = self.buffer_count % hp.Buffer_size
            batch_right = self.right_buffer[batch_indices]
            batch_left = self.left_buffer[batch_indices]
            batch_inputs = {'Right':batch_right,
                            'Left':batch_left}
            batch_targets = self.target_buffer[batch_indices]
            self.model.fit(
                x=batch_inputs,
                y=batch_targets,
                verbose=False,
                epochs=hp.Train_epoch,
            )
            if not self.total_steps % hp.Target_update:
                self.t_model.set_weights(self.model.get_weights())

        self.total_steps += 1
        self.current_steps += 1

    def save_model(self):
        """
        Return next save file number
        """
        self.save_count += 1
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
        self.model_dir = path.join(self.save_dir, str(self.save_count))
        self.model.save(self.model_dir)
        np.savez(
            path.join(self.model_dir,'buffer.npz'),
            Right=self.right_buffer,
            Left=self.left_buffer,
            Target=self.target_buffer
        )

        return self.save_count

    def evaluate(self, env, video_type):
        print('Evaluating...')
        done = False
        video_dir = path.join(self.model_dir, 'eval.{}'.format(video_type))
        eye_dir = path.join(self.model_dir, 'eval_eye.{}'.format(video_type))
        score_dir = path.join(self.model_dir, 'score.txt')
        if 'avi' in video_type :
            fcc = 'DIVX'
        elif 'mp4' in video_type:
            fcc = 'mp4v'
        else:
            raise TypeError('Wrong videotype')
        fourcc = cv2.VideoWriter_fourcc(*fcc)
        # Becareful : cv2 order of image size is (width, height)
        eye_out = cv2.VideoWriter(eye_dir, fourcc, 10, (205*5,50))
        out = cv2.VideoWriter(video_dir, fourcc, 10, env.image_size)
        eye_bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
        o = env.reset()
        score = 0
        loop = 0
        while not done :
            loop += 1
            if not loop % 100:
                print('Eval : {}step passed'.format(loop))
            a = self.act(o)
            o,r,done,i = env.step(a)
            if i['ate_apple']:
                score += 1
            #eye recording
            rt_eye = np.flip(o['Right'][:,-1,:],axis=0)
            lt_eye = o['Left'][:,-1,:]
            eye_img = np.concatenate((lt_eye,eye_bar,rt_eye))
            eye_img = np.broadcast_to(eye_img.reshape((1,205,1,3)),(50,205,5,3))
            eye_img = eye_img.reshape(50,205*5,3)
            eye_out.write(np.flip(eye_img, axis=-1))
            # This will turn image 90 degrees, but it does not make any difference,
            # so keep it this way to save computations
            out.write(np.flip(env.render('rgb'), axis=-1))
        out.release()
        eye_out.release()
        with open(score_dir, 'w') as f:
            f.write(str(score))
        print('Eval finished')
        return score

