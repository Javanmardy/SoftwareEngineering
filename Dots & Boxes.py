import multiprocessing
import random
import time
import copy
import tkinter as tk


class WorldModel(multiprocessing.Process):
    def __init__(
        self,
        state_queue_a,
        state_queue_b,
        action_queue,
        gui_queue,
        lock,
        starting_agent,
    ):
        super(WorldModel, self).__init__()
        self.state_queue_a = state_queue_a
        self.state_queue_b = state_queue_b
        self.action_queue = action_queue
        self.gui_queue = gui_queue
        self.lock = lock
        self.turn = starting_agent
        self.lines = {}
        self.boxes = {}
        self.init_game_state()

    def init_game_state(self):
        for i in range(3):
            for j in range(2):
                self.lines[((i, j), (i, j + 1))] = None
        for i in range(2):
            for j in range(3):
                self.lines[((i, j), (i + 1, j))] = None
        for i in range(2):
            for j in range(2):
                self.boxes[(i, j)] = None

    def run(self):
        while True:
            if all(owner is not None for owner in self.boxes.values()):
                scores = {"A": 0, "B": 0}
                for owner in self.boxes.values():
                    if owner is not None:
                        scores[owner] += 1
                self.gui_queue.put(("game_over", scores))
                break
            if self.turn == "A":
                self.state_queue_a.put((self.get_state(), self.turn))
            else:
                self.state_queue_b.put((self.get_state(), self.turn))
            try:
                action = self.action_queue.get(timeout=5)
            except:
                print(f"Agent {self.turn} did not respond in time.")
                break
            agent_id, line = action
            with self.lock:
                if agent_id != self.turn:
                    continue
                self.apply_action(action)
                completed_boxes = self.check_boxes_for_agent(agent_id)
                self.gui_queue.put(
                    ("update", copy.deepcopy(self.lines), copy.deepcopy(self.boxes))
                )
                time.sleep(1)
                if not completed_boxes:
                    self.turn = "B" if self.turn == "A" else "A"

    def get_state(self):
        with self.lock:
            return (copy.deepcopy(self.lines), copy.deepcopy(self.boxes))

    def apply_action(self, action):
        agent_id, line = action
        if self.lines.get(line) is None:
            self.lines[line] = agent_id

    def check_boxes_for_agent(self, agent_id):
        completed_boxes = []
        for box_pos in self.boxes:
            if self.boxes[box_pos] is None:
                sides = self.get_box_sides(box_pos)
                if all(self.lines.get(side) is not None for side in sides):
                    self.boxes[box_pos] = agent_id
                    completed_boxes.append(box_pos)
        return completed_boxes

    def get_box_sides(self, box_pos):
        i, j = box_pos
        top = ((i, j), (i, j + 1))
        bottom = ((i + 1, j), (i + 1, j + 1))
        left = ((i, j), (i + 1, j))
        right = ((i, j + 1), (i + 1, j + 1))
        return [top, bottom, left, right]


class Agent(multiprocessing.Process):
    def __init__(self, agent_id, state_queue, action_queue, lock):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.lock = lock
        self.max_depth = 3

    def run(self):
        while True:
            try:
                state, turn = self.state_queue.get(timeout=5)
                lines, boxes = state
                action = self.decide_action(lines, boxes, self.agent_id, turn)
                if action is None:
                    break
                self.action_queue.put((self.agent_id, action))
            except Exception as e:
                print(f"Exception in Agent {self.agent_id}: {e}")
                break

    def decide_action(self, lines, boxes, player, turn):
        best_score = float("-inf")
        best_moves = []
        available_lines = [line for line, owner in lines.items() if owner is None]
        random.shuffle(available_lines)

        for line in available_lines:
            new_lines = copy.deepcopy(lines)
            new_boxes = copy.deepcopy(boxes)
            new_lines[line] = player
            completed_boxes = self.update_boxes(new_boxes, new_lines, player)
            score = self.minimax(
                new_lines, new_boxes, self.max_depth - 1, False, player
            )
            if score > best_score:
                best_score = score
                best_moves = [line]
            elif score == best_score:
                best_moves.append(line)
        if best_moves:
            return random.choice(best_moves)
        else:
            return None

    def minimax(self, lines, boxes, depth, is_maximizing, player):
        if depth == 0 or all(owner is not None for owner in boxes.values()):
            return self.evaluate(boxes, player)

        available_lines = [line for line, owner in lines.items() if owner is None]
        if is_maximizing:
            max_eval = float("-inf")
            for line in available_lines:
                new_lines = copy.deepcopy(lines)
                new_boxes = copy.deepcopy(boxes)
                new_lines[line] = player
                completed_boxes = self.update_boxes(new_boxes, new_lines, player)
                if completed_boxes:
                    eval = self.minimax(new_lines, new_boxes, depth - 1, True, player)
                else:
                    eval = self.minimax(new_lines, new_boxes, depth - 1, False, player)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            opponent = "B" if player == "A" else "A"
            for line in available_lines:
                new_lines = copy.deepcopy(lines)
                new_boxes = copy.deepcopy(boxes)
                new_lines[line] = opponent
                completed_boxes = self.update_boxes(new_boxes, new_lines, opponent)
                if completed_boxes:
                    eval = self.minimax(new_lines, new_boxes, depth - 1, False, player)
                else:
                    eval = self.minimax(new_lines, new_boxes, depth - 1, True, player)
                min_eval = min(min_eval, eval)
            return min_eval

    def update_boxes(self, boxes, lines, player):
        completed_boxes = []
        for box_pos in boxes:
            if boxes[box_pos] is None:
                sides = self.get_box_sides(box_pos)
                if all(lines.get(side) is not None for side in sides):
                    boxes[box_pos] = player
                    completed_boxes.append(box_pos)
        return completed_boxes

    def evaluate(self, boxes, player):
        player_score = sum(1 for owner in boxes.values() if owner == player)
        opponent = "B" if player == "A" else "A"
        opponent_score = sum(1 for owner in boxes.values() if owner == opponent)
        return player_score - opponent_score

    def get_box_sides(self, box_pos):
        i, j = box_pos
        top = ((i, j), (i, j + 1))
        bottom = ((i + 1, j), (i + 1, j + 1))
        left = ((i, j), (i + 1, j))
        right = ((i, j + 1), (i + 1, j + 1))
        return [top, bottom, left, right]


class GUI:
    def __init__(self, gui_queue):
        self.gui_queue = gui_queue
        self.root = tk.Tk()
        self.root.title("Dots and Boxes")
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()
        self.line_ids = {}
        self.box_ids = {}
        self.create_board()
        self.root.after(100, self.update_gui)
        self.root.mainloop()

    def create_board(self):
        for i in range(3):
            for j in range(3):
                x = 100 + j * 100
                y = 100 + i * 100
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black")

    def update_gui(self):
        try:
            while True:
                message = self.gui_queue.get_nowait()
                if message[0] == "update":
                    lines, boxes = message[1], message[2]
                    self.draw_lines(lines)
                    self.fill_boxes(boxes)
                elif message[0] == "game_over":
                    scores = message[1]
                    winner = None
                    if scores["A"] > scores["B"]:
                        winner = "Agent A wins!"
                    elif scores["B"] > scores["A"]:
                        winner = "Agent B wins!"
                    else:
                        winner = "It's a tie!"
                    self.canvas.create_text(
                        200,
                        50,
                        text=f"Game Over! {winner}",
                        font=("Arial", 16),
                        fill="red",
                    )
        except:
            pass
        self.root.after(100, self.update_gui)

    def draw_lines(self, lines):
        for line, owner in lines.items():
            if owner and line not in self.line_ids:
                x1 = 100 + line[0][1] * 100
                y1 = 100 + line[0][0] * 100
                x2 = 100 + line[1][1] * 100
                y2 = 100 + line[1][0] * 100
                color = "blue" if owner == "A" else "green"
                self.line_ids[line] = self.canvas.create_line(
                    x1, y1, x2, y2, fill=color, width=2
                )

    def fill_boxes(self, boxes):
        for box_pos, owner in boxes.items():
            if owner and box_pos not in self.box_ids:
                x1 = 100 + box_pos[1] * 100 + 5
                y1 = 100 + box_pos[0] * 100 + 5
                x2 = x1 + 90
                y2 = y1 + 90
                color = "light blue" if owner == "A" else "light green"
                self.box_ids[box_pos] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color
                )
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=owner,
                    font=("Arial", 20),
                    fill="black",
                )


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    state_queue_a = manager.Queue()
    state_queue_b = manager.Queue()
    action_queue = manager.Queue()
    gui_queue = manager.Queue()
    lock = manager.Lock()

    starting_agent = random.choice(["A", "B"])
    print(f"Starting Agent: {starting_agent}")

    world_model = WorldModel(
        state_queue_a, state_queue_b, action_queue, gui_queue, lock, starting_agent
    )
    agent_a = Agent("A", state_queue_a, action_queue, lock)
    agent_b = Agent("B", state_queue_b, action_queue, lock)

    world_model.start()
    agent_a.start()
    agent_b.start()

    gui = GUI(gui_queue)

    world_model.join()
    agent_a.join()
    agent_b.join()
