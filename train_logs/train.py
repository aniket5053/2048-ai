import tkinter as tk
from tkinter import Frame, Label, CENTER, Button
import random
import logic
import constants as c
from ai import AdvancedAI2048
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

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
        self.load_model()

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

        self.episode_counter = Label(self, text="Episode: 0", font=("Verdana", 12))
        self.episode_counter.grid(row=c.GRID_LEN, column=0, columnspan=c.GRID_LEN)

        self.total_reward_display = Label(self, text="Total Reward: 0", font=("Verdana", 12))
        self.total_reward_display.grid(row=c.GRID_LEN + 1, column=0, columnspan=c.GRID_LEN)

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

    def start_training(self):
        self.is_training = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.after(c.MOVE_DELAY, self.ai_move)

    def stop_training(self):
        self.is_training = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def reset_game(self):
        self.stop_training()
        self.init_matrix()
        self.update_grid_cells()
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_counter.configure(text=f"Episode: {self.episode}")
        self.total_reward_display.configure(text=f"Total Reward: {self.total_reward}")
        self.update_graph()

    def ai_move(self):
        if not self.is_training:
            return

        state = self.ai.get_state(self.matrix)
        action = self.ai.choose_action(state)
        
        previous_matrix = [row[:] for row in self.matrix]
        self.matrix, move_done = self.commands[action](self.matrix)
        
        if move_done:
            self.matrix = logic.add_two(self.matrix)
            reward = sum(sum(row) for row in self.matrix) - sum(sum(row) for row in previous_matrix)
            next_state = self.ai.get_state(self.matrix)
            
            done = logic.game_state(self.matrix) == 'lose'
            self.ai.learn(state, action, reward, next_state, done)
            
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

    def load_model(self):
        try:
            checkpoint = torch.load('model.pth')
            self.ai.model.load_state_dict(checkpoint['model_state_dict'])
            self.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ai.exploration_rate = checkpoint['exploration_rate']
            self.episode_rewards = checkpoint['episode_rewards']
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found.")
