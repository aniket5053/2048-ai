# 2048 AI Game - Work in Progress

## Overview

This project is an AI-based implementation of the popular **2048 game**. The goal is to develop a smart AI agent capable of playing 2048 efficiently, using **Reinforcement Learning (Deep Q-Learning)**. The project also provides a graphical interface to visualize the game and track the AI's progress.

Currently, the AI is being trained to merge high-value tiles, prioritize specific strategies like keeping the highest tile in the corner, and improve gameplay over time through learning.

### Features
- **Classic 2048 Game**: Playable through a GUI built with Tkinter.
- **AI Integration**: The AI agent can play the game using a neural network and reinforcement learning techniques.
- **Learning Process Visualization**: Matplotlib is used to visualize rewards and track AI performance.
- **Save and Load Model**: Includes functionality for saving the model, optimizer state, exploration rate, and episode rewards, enabling the AI to resume learning from a previous session.

### Project Status
This project is a **work in progress**, with ongoing improvements in:
- Optimizing the AI’s strategy to merge tiles efficiently.
- Enhancing exploration decay for better long-term learning.
- Tuning reward functions for better tile positioning (e.g., keeping the highest tile in the bottom left corner).
- Improving AI decision-making through heuristic weights and training adjustments.

### Upcoming Features
- Further optimization of AI decision strategies.
- More refined reward functions to handle advanced game scenarios.
- Complete training for the AI model.
- Detailed documentation of the code and structure.
