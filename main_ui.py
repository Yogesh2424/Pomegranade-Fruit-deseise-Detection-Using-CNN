import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Disease Detector")
        # self.master.configure(background = "white")
        self.master.geometry("300x350")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Create button to choose image
        self.choose_image_button = tk.Button(self, text="Choose Image", command=self.choose_image, background="yellow")
        self.choose_image_button.pack()

        # Create label to show chosen image
        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Create button to run analysis function
        self.analyse_button = tk.Button(self, text="Analyse", command=self.analyse, background="yellow")
        self.analyse_button.pack()

        # Create label to show analysis output
        self.analysis_output_label = tk.Label(self)
        self.analysis_output_label.pack()

        # Create button to restart program
        self.restart_button = tk.Button(self, text="New", command=self.restart, background="yellow")
        self.restart_button.pack()

    def choose_image(self):
        # Open file dialog to choose image files
        file_path = filedialog.askopenfilename(initialdir="testing")
        self.filepath = file_path

        # Open and display chosen image
        self.image = Image.open(file_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo)

    def analyse(self):
        # Define function to be called when analyse button is pressed
        # Replace this with your own analysis function

        import cv2
        import os
        import numpy as np
        import tensorflow as tf

        path = "testing"
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

        path = os.path.join(path, self.filepath)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        check = [np.array(img)]

        import tflearn
        from tflearn.layers.conv import conv_2d, max_pool_2d
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.estimator import regression
        import tensorflow as tf

        # tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 5, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')
        model.load(MODEL_NAME)

        # input_image = check.reshape((1,) + check.shape)
        output = model.predict(check)
        tf.keras.backend.clear_session()

        def get_label(l):
            return ["Infected : Aanthracnose","Infected : Bacterial blight","Infected : Cercospora Fruit Spot","Infected : Fruit Rot","Healthy Fruit"][np.argmax(np.array(l))]

        def one_hot(l):
            index = np.argmax(l)
            one_hot = [0, 0, 0, 0, 0]
            one_hot[index] = 1
            return one_hot

        output = one_hot(output)
        output = get_label(output)


        analysis_output = output


        # Display analysis output
        self.analysis_output_label.configure(text=analysis_output)

    def restart(self):
        # Restart the program by creating a new instance of Application
        self.master.destroy()
        self.master = tk.Tk()
        self.master.geometry("300x350")
        app = Application(self.master)
        app.mainloop()

# Create and run application
root = tk.Tk()
app = Application(master=root)
app.mainloop()
