import tkinter as tk
from tkinter import Frame, Label, CENTER, Button
import random
import logic
import constants as c
from ai import AdvancedAI2048
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048 Advanced AI')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        self.ai = AdvancedAI2048()
        self.is_training = False
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []

        self.init_graph()

        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid(row=0, column=0, columnspan=c.GRID_LEN, padx=10, pady=10)

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(row=i, column=j, padx=c.GRID_PADDING, pady=c.GRID_PADDING)
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2
                )
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

        # Add episode counter
        self.episode_counter = Label(self, text="Episode: 0", font=("Verdana", 12))
        self.episode_counter.grid(row=c.GRID_LEN, column=0, columnspan=c.GRID_LEN)

        # Add total reward display
        self.total_reward_display = Label(self, text="Total Reward: 0", font=("Verdana", 12))
        self.total_reward_display.grid(row=c.GRID_LEN + 1, column=0, columnspan=c.GRID_LEN)

        # Add control buttons
        self.start_button = Button(self, text="Start Training", command=self.start_training)
        self.start_button.grid(row=c.GRID_LEN + 2, column=0, columnspan=2)

        self.stop_button = Button(self, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_button.grid(row=c.GRID_LEN + 2, column=2, columnspan=2)

        self.reset_button = Button(self, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=c.GRID_LEN + 3, column=0, columnspan=c.GRID_LEN)

    def init_matrix(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []

    def init_graph(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("AI Learning Progress")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.line, = self.ax.plot([], [], 'b-')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=c.GRID_LEN, rowspan=c.GRID_LEN, padx=10, pady=10)

    def update_graph(self):
        episodes = list(range(1, len(self.episode_rewards) + 1))
        self.line.set_data(episodes, self.episode_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        if key == c.KEY_QUIT:
            exit()
        if not self.is_training and key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.update_grid_cells()
                done = False
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def start_training(self):
        self.is_training = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.after(c.MOVE_DELAY, self.ai_move)

    def stop_training(self):
        self.is_training = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def save_rewards(self):
        with open("episode_rewards.txt", "w") as f:
            for episode, reward in enumerate(self.episode_rewards):
                f.write(f"{episode},{reward}\n")
        print("Rewards saved to episode_rewards.txt")

    def reset_game(self):
        self.stop_training()
        self.init_matrix()
        self.update_grid_cells()
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.save_rewards()  # Save rewards at reset
        self.episode_counter.configure(text=f"Episode: {self.episode}")
        self.total_reward_display.configure(text=f"Total Reward: {self.total_reward}")
        self.update_graph()

    def ai_move(self):
        if not self.is_training:
            return

        state = self.ai.get_state(self.matrix)
        current_score = sum(sum(row) for row in self.matrix)  # Calculate current score
        action = self.ai.choose_action(state)

        previous_matrix = [row[:] for row in self.matrix]

        # Create a mapping from action strings to their indices
        action_to_index = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
        
        # Get the numeric index for the chosen action
        action_index = action_to_index[action]


        self.matrix, move_done = self.commands[action](self.matrix)

        if move_done:
            self.matrix = logic.add_two(self.matrix)
            reward = sum(sum(row) for row in self.matrix) - sum(sum(row) for row in previous_matrix)
            next_state = self.ai.get_state(self.matrix)

            done = logic.game_state(self.matrix) == 'lose'
            
            # Pass the numeric action index instead of the string action
            self.ai.learn(state, action_index, reward, next_state, done)

            self.update_grid_cells()
            self.total_reward += reward
            self.total_reward_display.configure(text=f"Total Reward: {self.total_reward}")

            if done:
                self.episode += 1
                self.episode_counter.configure(text=f"Episode: {self.episode}")
                self.episode_rewards.append(self.total_reward)
                self.update_graph()
                self.init_matrix()
                self.total_reward = 0

            self.ai.decay_exploration()

        self.after(c.MOVE_DELAY, self.ai_move)


game_grid = GameGrid()