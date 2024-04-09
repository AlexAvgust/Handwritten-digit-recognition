from PIL import ImageTk, ImageGrab, Image
import PIL.Image
import io
import os
from tkinter import *
import tkinter as tk
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

window = Tk()
window.configure(background='darkgray')
window.title("Digit recognition")
window.geometry('1120x820')
window.iconbitmap('./icon.ico')
window.resizable(0, 0)




model_path = f'{os.getcwd()}/training/mnist_model.keras'
model = load_model(model_path)

def destroy_widget(widget): 
    widget.destroy()


def predict_digit():
    global no,no1
    ps = canvas.postscript(colormode='color')
    im1= PIL.Image.open(io.BytesIO(ps.encode('utf8')))
    img= im1.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    #predictions 
    res = model.predict([img])[0]
    pred = np.argmax(res)
    print(pred)
    print(f'Predicted Digit is: {res}')
    acc = max(res)
    
    no = tk.Label(window,text=f'Predicted Digit is: {pred}', 
                  width=50, height=1, fg='white', 
                  bg='midnightblue', font=('times',16, 'bold'))
    no.place(x=460, y=380)
    no1 = tk.Label(window, text=f'Prediction Accuracy is: {acc}',
                   width=50, height=1,
                   fg="white", bg="red",
                   font=('times', 16, ' bold '))
    no1.place(x=460,y=415)
    
 
    
    clear_button.configure(state=NORMAL)



predict_button = tk.Button(window,text='Predict digit',
                           state=DISABLED,command = predict_digit,
                           width = 15,borderwidth=0,bg = 'midnightblue',
                           fg = 'white',font = ('times',18,'bold') )
    
predict_button.place(x=60,y=90)

def clear_digit():
    predict_button.configure(state=DISABLED)
    canvas.delete('all')
    try:
        no.destroy()
        no1.destroy()
    except:
        pass

clear_button = Button(window,text = 'Clear Digit',
                      state=DISABLED,command = clear_digit,
                      width = 15,borderwidth=0,bg = 'midnightblue',
                      fg = 'white',font = ('times',18,'bold'))
clear_button.place(x=60, y=125)

def draw_digit(event):
    x = event.x
    y = event.y
    r = 10
    canvas.create_oval(x-r, y-r, x + r, y + r, fill='black')
    predict_button.configure(state=NORMAL)
    
    
canvas = tk.Canvas(window,width=405,
                   height=280,highlightthickness=1,
                   highlightbackground='midnightblue',cursor='pencil')
canvas.grid(row=0, column=0, pady=2, sticky=W,)
canvas.place(x=460,y=90)
canvas.bind("<B1-Motion>", draw_digit)
window.mainloop()