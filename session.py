import os, random, torch, time
import matplotlib.pyplot as plt

class TrainingSession:
    def __init__(self, model_name):
        self.model_name = model_name.replace(" ", "_")
        self.session_name = model_name + "_"  + str(int(time.time())) + "_" + str(random.randint(0, 1000000))
        self.session_path = os.path.join("sessions", self.session_name)
        
        # Create session folder
        if not os.path.exists(self.session_path):
            os.makedirs(self.session_path)

        self.start_time = None
        self.step = 0

    def info(self, text):
        with open(os.path.join(self.session_path, "log.txt"), "a") as f:
            f.write(text + "\n")

    def save(self, state_dict):
        if not os.path.exists(self.session_path):
            os.mkdir(self.session_path)
        
        torch.save(state_dict, os.path.join(self.session_path, "model_state_dict.pt"))
    
    def log(self, loss, step=None, log_type="train"):
        self.step += 1

        if self.start_time is None:
            self.start_time = time.time()
        
        delta = round(time.time() - self.start_time, 3)
        
        if step is None:
            step = self.step

        with open(os.path.join(self.session_path, log_type + ".txt"), "a") as f:
            f.write(f"{step}\t{delta}\t{loss}\n")

    def plot(self, log_type="train", roll_window=1):
        with open(os.path.join(self.session_path, log_type + ".txt"), "r") as f:
            lines = f.readlines()
        
        x = []
        y = []
        for i, line in enumerate(lines):
            line = line.split("\t")
            x.append(float(i))
            y.append(float(line[2]))
        
        # Rolling average, if accumated batch size results are "noisy"
        n = roll_window
        y = [sum(y[i:i+n])/n for i in range(len(y)-n)]
        x = x[n//2:-n//2]
        
        plt.clf()
        plt.title(self.model_name)
        plt.plot(x, y)
        plt.savefig(os.path.join(self.session_path, f"{log_type}_plot.png"))