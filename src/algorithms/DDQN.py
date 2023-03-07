'''DDQN Implementation'''

class DDQN:
    '''Double Deep Q Learning Implementation'''


    class Trainer:
        '''Class for training a DDQN implementation using a generic game interface'''

        #number of frames to run
        NUM_FRAMES = 10000
        #max memory stored for exp replay
        MAX_MEMORY = int(NUM_FRAMES/5)
        #initial population of memory using random policy
        INIT_MEMORY = int(MAX_MEMORY/4)
        #max iterations per run
        MAX_ITERATIONS = 100
        #update interval to use target network
        TARGET_C = 3000


        def __init__(self, num_frames = NUM_FRAMES, model_file_name = 'model'):
            if num_frames is not None:
                self.NUM_FRAMES = num_frames
                #max memory stored for exp replay
                self.MAX_MEMORY = int(self.NUM_FRAMES/10)

                #initial population of memory using random policy
                self.INIT_MEMORY = int(self.MAX_MEMORY/30)

                #max iterations per run
                self.MAX_ITERATIONS = 100

                #update interval to use target network
                # self.TARGET_C = int(self.NUM_FRAMES/300)
            if model_file_name is not None:
                self.model_file_name = model_file_name
            else:
                self.model_file_name = 'model'


            # self.model_controller = ModelController(self)
            # self.game = Game(self.model_controller)
            # # self.memory : deque[MemoryItem] = deque(maxlen=90000)
            # self.memory = []
            # try:
            #     self.model = ModelInterface(load = True, model_file = model_file_name)
            #     self.target_model = ModelInterface(load = True, model_file = model_file_name)
            # except Exception:
            #     self.model = ModelInterface(load = False, model_file = model_file_name)
            #     self.target_model = ModelInterface(load = False, model_file = model_file_name)

            # self.filter_threshold = 0.9

        def prefill_mem(self):
            '''Prefill memory'''
            print('Prefilling Memory')
            with Bar('Processing', max = self.INIT_MEMORY) as bar:
                while len(self.memory) < self.INIT_MEMORY:
                    # reset game
                    self.model_controller.initialise()
                    self.game = Game(self.model_controller)
                    state = self.game.generate_input_vector()
                    buffer_input = [
                        np.zeros(INPUT_DIMENSIONS)
                    ] * PREV_FRAMES
                    buffer_input.append(state)
                    buffer_input = buffer_input[-1 * PREV_FRAMES : ]
                    # state = np.array(buffer_input).reshape(-1)
                    state = tuple(buffer_input)
                    done = False

                    while not done:
                        game_action = self.model.get_random_action()
                        done, reward = self.model_controller.process(self.game, self.model.convert_action_to_game_action(game_action))
                        new_state = self.game.generate_input_vector()
                        buffer_input.append(new_state)
                        buffer_input = buffer_input[-1 * PREV_FRAMES : ]
                        # new_state = np.array(buffer_input).reshape(-1)
                        new_state = tuple(buffer_input)
                        # self.memory.append((np.copy(state),np.copy(new_state), game_action,reward,done))
                        self.memory.append(MemoryItem(state,new_state,game_action,reward,done))
                        bar.next()
                        state = new_state

        def _filter_game_mem(self, game_memory : list) -> list:
            '''filter game memory and adjust thresholds, then add the filtered game memory'''
            new_mem = []
            count_reward_low = 0
            count_reward_total = 0.001
            for state,new_state, action, reward, done in game_memory:
                if (reward < self.LOW_REWARD_THRESHOLD or np.random.random() > self.filter_threshold):
                    if reward < self.LOW_REWARD_THRESHOLD:
                        count_reward_low += 1
                    count_reward_total += 1
                    new_mem.append((state,new_state,action,reward,done))
            ratio = count_reward_low / count_reward_total
            if ratio < self.REWARD_LOW_TO_REWARD_RATIO:
                self.filter_threshold /= 0.95
            if ratio >  self.REWARD_LOW_TO_REWARD_RATIO:
                self.filter_threshold  *= 0.95
            return new_mem

        def run_frames(self):
            '''Run through the frames to train the model'''
            total_frames = 0
            #epsilon for choosing action
            eps = 1
            #minimum eps
            eps_min = 0.2
            #eps linear decay for first 10% of run
            eps_linear_decay = (eps-eps_min)/(self.NUM_FRAMES/100)
            #discount factor for future utility
            discount_factor = 0.9
            batch_size = 128
            num_updates = 0
            frame_mean = 0
            print('Running Frames')
            game_mem = []
            frames = []
            num_games = 0
            with Bar('Processing', max = self.NUM_FRAMES) as progress_bar:
                while total_frames < self.NUM_FRAMES:
                    num_games += 1
                    # reset game
                    self.model_controller.initialise()
                    self.game = Game(self.model_controller)
                    state = self.game.generate_input_vector()
                    # frames = 0

                    buffer_input = [
                        np.zeros(INPUT_DIMENSIONS)
                    ] * PREV_FRAMES
                    buffer_input.append(state)
                    buffer_input = buffer_input[-1 * PREV_FRAMES : ]
                    # state = np.array(buffer_input).reshape(-1)
                    state = tuple(buffer_input)
                    #playing through this round
                    done = False
                    game_frames = 0
                    while not done:
                        game_frames += 1

                        action = epsilon_greedy(eps, self.model, state)


                        done, reward = self.model_controller.process(
                            self.game, self.model.convert_action_to_game_action(action))
                        new_state = self.game.generate_input_vector()
                        buffer_input.append(new_state)
                        buffer_input = buffer_input[-1 * PREV_FRAMES : ]
                        # new_state = np.array(buffer_input).reshape(-1)
                        new_state = tuple(buffer_input)

                        if random.random() > 0.8 or done:
                            total_frames += 1
                            progress_bar.next()
                            game_mem.append(
                                MemoryItem(state,new_state,action,reward,done))
                            # MemoryItem(np.copy(state),np.copy(new_state),action,reward,done))
                        if done:
                            break

                        #update state
                        state = new_state

                        #decay epsilon
                        eps -= eps_linear_decay
                        eps = max(eps, eps_min)

                    if len(game_mem) > batch_size:
                        self.memory += game_mem
                        game_mem.clear()
                        num_updates += batch_size
                        # print('num_updates',num_updates, self.TARGET_C)
                        if num_updates > self.TARGET_C:
                            num_updates = 0
                            # print('Experience')
                            self.target_model.model.set_weights(self.model.model.get_weights())
                            self.model.model.save(self.model_file_name)
                        experience_replay(self.memory, self.model, self.target_model, discount_factor, batch_size)

                    if total_frames > self.NUM_FRAMES:
                        break
                frame_mean = frame_mean*0.95 + 0.05*game_frames
                frames.append(game_frames)
                print(' Frames', frame_mean, eps)

                if len(self.memory) > 90000:
                    self.memory = self.memory[:90000]

                if num_games % 10 == 0:
                    with open('frames.json','w',encoding='utf8') as file:
                        file.write(json.dumps(frames))
                # max_score = game_mem[-1][3]
                # shift rewards to be the time survived from an action
                # game_mem = [
                #     MemoryItem(state,new_state, action,max_score - reward, done)
                #     for state,new_state,action,reward,done in game_mem
                # ]


                # filter game memory and adjust thresholds
                # if self.USE_FILTER:
                #     game_mem = self._append_game_memory_with_filter(game_mem)

                # Convert to rewards
                # game_mem = reward_if_survive_frame(game_mem)

                # self.memory += game_mem

                # total_frames += frames

                    # reset memory
                    # self.memory = []
        with open('frames.json','w',encoding='utf8') as file:
            file.write(json.dumps(frames))
