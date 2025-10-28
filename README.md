[Neural_Network.py](https://github.com/user-attachments/files/23193424/Neural_Network.py)[Uploading Nimport tensorflow as tf
import keras
#import sklearn
import datetime
import matplotlib.pyplot as plt
import numpy as np
import socha
import socha.api.protocol.protocol
import typing
import random
import pickle
import os
import time

class nn_model:
    def __init__(self, is_main: bool):
        t1 = time.time()
        if is_main:
            
            self.memory : list[tuple[np.ndarray | list[list[None]], tuple[list[list[str]], int], int, float, bool]]= []
            try:
                with open("C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\memory.pkl", "rb") as f:
                    self.memory = pickle.load(f)
                print("Memory eingelesen")
            except:
                print("Datei nicht vorhanden")
            print(f"Anfangslänge Memory: {len(self.memory)}")
        self.neurons = 140
        self.output_neurons = 128
        self.gamma = 0.99
        self.epsilon = 0.35
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.25
        self.batch_size = 512
        self.max_memory = 60000
        self.episodes = 5
        self.last_state = [[None]]
        self.loss_history = []
        self.is_main = is_main
        self.own_team = None
        
        self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        self.nn_model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding = "same", input_shape=(10,10,5)),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding = "same"),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding = "same"),
            keras.layers.Flatten(),
            keras.layers.Dense(self.neurons*2, activation='relu'),
            keras.layers.Dense(self.neurons, activation='relu'),
            keras.layers.Dense(self.output_neurons, activation='linear')
            ])
        self.nn_model.compile(optimizer=self.optimizer, loss=keras.losses.Huber())
        #print(self.nn_model.summary())
        try:
            self.nn_model = keras.models.load_model("C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\new_saved_model", compile=True) 
            print("Modell geladen")
        except:
            print('Noch kein Modell vorhanden')
        #print(self.nn_model.summary())
        t2 = time.time()
        print(f"Zeit zum einlesen des Memorys: {t2-t1} Sekunden")
        

    def vorhersage(self, game_state: socha.GameState) -> socha.Move:
        #eigenes Team herausfinden, je nach aktuellem Zug
        if self.own_team is None:
            self.own_team_enum = socha.RulesEngine.get_team_on_turn(game_state.turn)
            self.own_team = str(socha.RulesEngine.get_team_on_turn(game_state.turn)) == str(socha.TeamEnum.Two)
        print(f"Team Enum: {self.own_team_enum}")
        #print(f"Team: {self.own_team}")
        self.game_state = game_state
        self.X = self.reshape_gamestate(game_state)
        #print(self.X)
        self.q_values = list(self.nn_model.predict(self.X, verbose=0)[0]) #type: ignore
        #print(self.q_values)
        self.q_values_moves = self.q_values_to_move(gamestate=game_state, q_values=self.q_values)
        self.possible_q_values : list[tuple[socha.Move, float, int]]= []
        self.possible_moves = [str(move) for move in game_state.possible_moves()]
        #print(self.possible_moves)
        #Die legalen q-Werte sortieren
        for move in self.q_values_moves:
            if str(move[0]) in self.possible_moves:
                self.possible_q_values.append(move)
            else:
                self.possible_q_values.append((move[0], float('-inf'), move[2])) 
        #größten q-Wert mit Zug suchen
        self.max_tuple = (socha.Move(socha.Coordinate(1,1), socha.Direction.Up), float('-inf'), int(1)) #max_tuple mit Platzhalter füllen
        for tup in self.possible_q_values:
            if self.max_tuple is None or tup[1] > self.max_tuple[1]:  
                self.max_tuple = tup
        
        #Aktion wählen
        if np.random.rand() < self.epsilon:
            self.possible_q_values_random : list[tuple[socha.Move, float, int]] = []
            for move in self.possible_q_values:
                if move[1] != float('-inf'):
                    self.possible_q_values_random.append(move)
            self.max_tuple = random.choice(self.possible_q_values_random)
        
        self.action = self.max_tuple[0]
        if self.is_main:
            print(f"Länge memory vor dem Trainieren: {len(self.memory)}")
            self.train()
            if isinstance(self.last_state, np.ndarray):
                self.update_memory(game_state)
        self.last_state = self.X
        self.last_action = self.max_tuple[2]  
        
        return self.action

    def update_memory(self, gamestate: socha.GameState, roomMessage: typing.Optional[socha.api.protocol.protocol.Result] = None) -> None:
        #print(f"Reward: {self.reward_function(gamestate, (roomMessage if roomMessage else None))}")
        print(f"Länge memory vor dem Update: {len(self.memory)}")
        self.memory.append((self.last_state, self.gamestate_to_array(gamestate), self.last_action, self.reward_function(gamestate, (roomMessage if roomMessage else None)), True if roomMessage else False))
        if len(self.memory) > self.max_memory:
            self.memory= self.memory[4000:]
        print(f"Länge memory: {len(self.memory)}")

    def train(self):
        # Training mit Replay Memory
        if len(self.memory) > 5000:
            print("Es fängt an zu trainieren!")
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.vstack([state for state, _, _, _, _ in minibatch])
            
            next_states = np.vstack([self.reshape_gamestate(self.gamestate_array_to_gamestate(next_state)) for _, next_state, _, _, _ in minibatch])
            
            targets = self.nn_model.predict(states, verbose=0)
            
            next_qs = self.nn_model.predict(next_states, verbose=0)
            
            
            
            for i, (_, gamestate, action, reward, game_finished) in enumerate(minibatch): 
                #target = self.nn_model.predict(last_state, verbose=0) #type: ignore
                #t = self.nn_model.predict(X, verbose=0) #Wie ist das target bei gamestate:is_over = True? #type: ignore
                #print(f"X: {X}")
                #print(f"t: {t}")
                gamestate= self.gamestate_array_to_gamestate(gamestate)
                next_q = next_qs[i]
                next_q_moves = self.q_values_to_move(gamestate=gamestate, q_values=next_q)
                possible_next_q_values : list[float]= []
                possible_next_q_moves = [str(move) for move in gamestate.possible_moves()]
                #print(self.possible_moves)
                #Die legalen q-Werte sortieren
                for move in next_q_moves:
                    if str(move[0]) in possible_next_q_moves:
                        possible_next_q_values.append(move[1])
                    else:
                        possible_next_q_values.append(float('-inf'))
                max_q = np.amax(possible_next_q_values) if len(possible_next_q_values) > 0 else float(0)
                targets[i][action] = reward + ((self.gamma * max_q) if not game_finished else 0)
            
            self.history = self.nn_model.fit(states, targets, epochs=1) #type: ignore
            for l in self.history.history['loss']:
                self.loss_history.append(l) 
                #Das Model speichern
           # self.nn_model.save("C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\new_saved_model", save_format='tf') #type: ignore
           # print("Das Model wurde erfolgreich gespeichert")
            print(self.game_state.turn)
            
            # Epsilon reduzieren
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            # Backups des aktuellen Trainingszustands erstellen
            if len(self.memory) % 5000 == 0:
                time = datetime.datetime.now().strftime('%d. %b, %Hh-%Mm-%Ss')
                self.nn_model.save(f"C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\Backups_Model\\model_checkpoint {time}\\model", save_format='tf') #type: ignore
                self.append_numpy("C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\loss_history.npy", self.loss_history, f"C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\Backups_Model\\model_checkpoint {time}\\previous_loss_history.npy")
                with open(f"C:\\Users\\Anwender\\Desktop\\Frederiks Ordner\\Programmieren\\Softwarechallenge\\Runde 2025-26\\Backups_Model\\model_checkpoint {time}\\previous_memory.pkl", "wb") as f:
                    pickle.dump(self.memory, f)
                print(f"Model und loss history und memory wurde erfolgreich gesichert, Länge memory: {len(self.memory)}")
            
            
        
        
        
    @staticmethod
    def plot_loss(loss_history = []):
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Training Loss', color='blue')
        plt.xlabel('Trainingsschritte')
        plt.ylabel('Loss')
        plt.title('Verlauf des Trainings-Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def reward_function(self, game_state: socha.GameState, result: typing.Optional[socha.api.protocol.protocol.Result] = None) -> float:
        reward = float(0)
        turn = game_state.turn
        if result is not None:
            #Belohnung für Sieg oder Niederlage und Anzahl an Zügen
            winner = result.winner.team
            if winner is None:
                reward = 0
            elif winner == self.own_team:
                reward = 30 *(1/turn)
            elif winner != self.own_team:
                reward = -30 *(1/turn)

        elif result is None:
            #Belohnungen für Veränderungen im Spiel
            if isinstance(self.last_state, np.ndarray):
                game_state_num_cur = self.reshape_gamestate(game_state)
                game_state_num_prev = self.last_state
                
                ##Belohnung für Fische fressen
                #reward += (self.fish_points_left_in_gamestate_num(game_state_num_cur, True) - self.fish_points_left_in_gamestate_num(game_state_num_prev, True)) * 0.05
                #reward -= (self.fish_points_left_in_gamestate_num(game_state_num_cur, False) - self.fish_points_left_in_gamestate_num(game_state_num_prev, False)) * 0.05
#
                ##Belohnung für Veränderungen der größten Schwarmgröße
                #reward += (self.largest_swarm_size_in_gamestate_num(game_state_num_cur, True) - self.largest_swarm_size_in_gamestate_num(game_state_num_prev, True)) * 0.02
                #reward -= (self.largest_swarm_size_in_gamestate_num(game_state_num_cur, False) - self.largest_swarm_size_in_gamestate_num(game_state_num_prev, False)) * 0.02

        self.reward = np.clip(reward, -1, 1)
        return self.reward
    
    def q_values_to_move(self, q_values : list[float], gamestate : socha.GameState) -> list[tuple[socha.Move, float, int]]:
        '''Fügt jedem Q-Wert den entsprechenden Move hinzu'''
        q_values_moves = []
        coords_with_fish = []
        for x_coord in range(10):
            for y_coord in range(10):
                if str(gamestate.board.get_field(socha.Coordinate(x=x_coord, y=y_coord)).get_team()) == str(socha.RulesEngine.get_team_on_turn(gamestate.turn)): #type: ignore
                    coords_with_fish.append(socha.Coordinate(x=x_coord, y=y_coord))
        i = 0
        #print(coords_with_fish)
        for coord in coords_with_fish:
            for direction in socha.Direction.all_directions():
                move = socha.Move(coord, direction)
                q_value = q_values[i]
                q_values_moves.append((move, q_value, i))
                #print(i)
                i += 1
        return q_values_moves
                
    
    def reshape_gamestate(self, game_state : socha.GameState) -> np.ndarray:
        '''Wandelt das GameState Objekt in ein Array um, welches als Input für das Neuronale Netz genutzt werden kann'''
        turn = game_state.turn
        normalized_turn = turn / 60
        if self.own_team == 1: #Eigene Fische immer [1,0,0,x] und Gegner [0,1,0,x]
            mappingFields = {str(socha.FieldType.Empty): [0,0,0,0, normalized_turn], 
                             str(socha.FieldType.OneS): [0,1,0,1, normalized_turn],
                             str(socha.FieldType.OneM): [0,1,0,2, normalized_turn],
                             str(socha.FieldType.OneL): [0,1,0,3, normalized_turn],
                             str(socha.FieldType.TwoS): [1,0,0,1, normalized_turn],
                             str(socha.FieldType.TwoM): [1,0,0,2, normalized_turn],
                             str(socha.FieldType.TwoL): [1,0,0,3, normalized_turn],
                             str(socha.FieldType.Squid): [0,0,1,0, normalized_turn]}
        else:
            mappingFields = {str(socha.FieldType.Empty): [0,0,0,0, normalized_turn], 
                             str(socha.FieldType.OneS): [1,0,0,1, normalized_turn],
                             str(socha.FieldType.OneM): [1,0,0,2, normalized_turn],
                             str(socha.FieldType.OneL): [1,0,0,3, normalized_turn],
                             str(socha.FieldType.TwoS): [0,1,0,1, normalized_turn],
                             str(socha.FieldType.TwoM): [0,1,0,2, normalized_turn],
                             str(socha.FieldType.TwoL): [0,1,0,3, normalized_turn],
                             str(socha.FieldType.Squid): [0,0,1,0, normalized_turn]}
        
        game_state_num = [[mappingFields[str(Field)] for Field in row] for row in game_state.board.map] 
        
        game_state_num = np.array(game_state_num) #In Numpy Array umwandeln
        game_state_num = np.expand_dims(game_state_num, axis=0)
        
        return game_state_num

    def fish_points_left_in_gamestate_num(self, game_state_num: np.ndarray, team: bool) -> int:
        """Zählt die Punkte der Fische im gegebenen game_state_num für das gegebene Team
        Args:
            game_state_num (np.ndarray): Das GameState-Array, wie es vom reshape_gamestate zurückgegeben wird
            team (bool): Das Team, für das die Punkte gezählt werden sollen (True für eigenes Team, False für Gegnerteam)"""
        points = 0
        for row in game_state_num[0]:
            for field in row:
                if field[0] == team:
                    points += field[3] 
        return points
    
    def largest_swarm_size_in_gamestate_num(self, game_state_num: np.ndarray, team: bool) -> int:
        """Findet die größte Schwarmgröße im gegebenen game_state_num für das gegebene Team
        Args:
            game_state_num (np.ndarray): Das GameState-Array, wie es vom reshape_gamestate zurückgegeben wird
            team (bool): Das Team, für das die größte Schwarmgröße gefunden werden soll (True für eigenes Team, False für Gegnerteam)"""
        visited = np.zeros((10, 10), dtype=bool)
        largest_size = 0
        # mögliche Berührungen (8 Richtungen)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

        # Tiefensuche (DFS) zur Bestimmung des Schwarmgewichts
        def dfs(x: int, y: int) -> int:
            if (x < 0 or x >= 10 or y < 0 or y >= 10 or 
                visited[x,y] or game_state_num[0][y][x][0] != team):
                return 0
            visited[x, y] = True
            size = game_state_num[0][y][x][3]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 10 and 0 <= ny < 10:
                    size += dfs(nx, ny)
            return size

        #DFS für jedes Feld starten
        for y in range(10):
            for x in range(10):
                if not visited[x, y] and game_state_num[0][y][x][0] == team:
                    swarm_size = dfs(x, y)
                    largest_size = max(largest_size, swarm_size)

        return largest_size
    
    def append_numpy(self, read_filename, new_values, write_filename=None):
        new_values = np.array(new_values)

        if os.path.exists(read_filename):
            # Bestehende Werte laden
            old_values = np.load(read_filename)
            old_values = np.atleast_1d(old_values)
            # Neue anhängen
            if len(old_values) != 0:
                all_values = np.concatenate([old_values, new_values])
            else:
                all_values = new_values
        else:
            all_values = new_values

        # Alles zurückspeichern
        if write_filename is None:
            write_filename = read_filename
        
        np.save(write_filename, all_values)
    
    def gamestate_array_to_gamestate(self, array: tuple[list[list[str]], int]) -> socha.GameState:
        """Wandelt ein abgespeichertes Gamestate Array zurück in ein komplexes GameState-Objekt."""
        board_map = []
        string_to_field = {
        "OneS": socha.FieldType.OneS,
        "OneM": socha.FieldType.OneM,
        "OneL": socha.FieldType.OneL,
        "TwoS": socha.FieldType.TwoS,
        "TwoM": socha.FieldType.TwoM,
        "TwoL": socha.FieldType.TwoL,
        "Squid": socha.FieldType.Squid,
        "Empty": socha.FieldType.Empty
        }

        for row in array[0]:
            board_row = []
            for field_name in row:
                board_row.append(string_to_field[field_name])
            board_map.append(board_row)
        
        board = socha.Board(board_map)
        turn = array[1]
        game_state = socha.GameState(board=board, turn=turn, last_move=None)
        return game_state

    def gamestate_to_array(self, gamestate: socha.GameState) -> tuple[list[list[str]], int]:
        """Wandelt das komplexe GameState-Objekt in ein einfacheres Array um, das von Pickle gespeichert werden kann."""
        board_array = []
        for row in gamestate.board.map:
            row_array = []
            for field in row:
                row_array.append(field.__repr__()) 
            board_array.append(row_array)
        turn = gamestate.turn
        return (board_array, turn)
    
    eural_Network.py…]()
