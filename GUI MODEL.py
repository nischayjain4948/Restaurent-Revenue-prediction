import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
import tkinter as tk
model = pd.read_pickle('RestaurantRevenuePrediction')
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tkinter import*
app=tk.Tk()
app.geometry('500x500')
app.title("R_R_system")
app.iconbitmap(r'D:\Restaurant Revenu\logo.ico')
Photo = PhotoImage(file='D:\Restaurant Revenu\pic.png')
photo_Label = Label(image=Photo)
photo_Label.pack()



Checkbutton1 = tk.IntVar()  
Checkbutton2 = tk.IntVar()
# tk.Label(app,width=15,text="P19",bg="gold",fg="black",font="bold").place(x=80,y=30)
tk.Label(app,width=15,text="Enter  P29",bg="#c0c0c0",fg="black",font='arial 10 underline').place(x=80,y=80)
tk.Label(app,width=15,text="Enter  P27",bg="#c0c0c0",fg="black",font='arial 10 underline').place(x=80,y=110)
tk.Label(app,width=15,text="Enter Date",bg="#c0c0c0",fg="black",font='arial 10 underline').place(x=80,y=140)
tk.Label(app,width=15,text="Enter  Year",bg="#c0c0c0",fg="black",font='arial 10 underline').place(x=80,y=170)
tk.Label(app,width=15,text="City",bg="#c0c0c0",fg="black",font='arial 10 underline').place(x=80,y=200)
tk.Label(app,width=10,text="Revenue :",fg="black",bg="#c0c0c0",font=("Helvetica", 15,"bold"),relief="groove").place(x=80,y=310)
tk.Checkbutton(app, text = "Izmir",variable = Checkbutton1,onvalue = 1,offvalue = 0,height =1,width =5,bg="#c0c0c0").place(x=290,y=200)
tk.Checkbutton(app, text = "Istanbul",variable = Checkbutton2,onvalue = 1,offvalue = 0,height = 1,width =5,bg="#c0c0c0").place(x=370,y=200)
p29=tk.Variable(app)
p27=tk.Variable(app)
date=tk.Variable(app)
year=tk.Variable(app)
tk.Entry(app,width=20,textvariable=p29,bg='#ffffff',relief="groove").place(x=290,y=80)
tk.Entry(app,width=20,textvariable=p27,bg='#ffffff',relief="groove").place(x=290,y=110)
tk.Entry(app,width=20,textvariable=date,bg='#ffffff',relief="groove").place(x=290,y=140)
tk.Entry(app,width=20,textvariable=year,bg='#ffffff',relief="groove").place(x=290,y=170)
predict_var = tk.Variable(app)
tk.Label(app,textvariable=predict_var,fg="black",bg="yellow",font=("Helvetica", 15,"bold")).place(x=260,y=310)
def predict():
    query = pd.DataFrame({
    'P29':[p29.get()],
    'Year':[year.get()],
    'P27':[p27.get()],
    'İzmir': [Checkbutton1.get()],
    'İstanbul': [Checkbutton2.get()],
    'Date': [date.get()]
})
    e=model.predict(query)[0]
    predict_var.set("{0:.1f}$".format(e))
    p29.set('')
    year.set('')
    p27.set('')
    date.set('')
tk.Button(app,text="Predict",fg="black",bg="green",font='arial 15 underline',relief="groove",command=predict).place(x=200,y=250) 
app.mainloop()
