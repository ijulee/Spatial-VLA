import tkinter as tk
import inspect
from unitTests import UnitTest 
#START UP GUI HERE

TESTS = [fn for _, fn in inspect.getmembers(UnitTest, predicate=inspect.isfunction)]
def on_run():
    selected = next((fn for fn in TESTS if fn.__name__ == selected_var.get()), None)
    if selected:
        selected()


root = tk.Tk()
root.title("Spatial-VLA unitTest selector")

selected_var = tk.StringVar(value=None)

for fn in TESTS:
    tk.Radiobutton(root, text=fn.__name__, variable=selected_var, value=fn.__name__).pack(anchor="w", padx=10, pady=2)

tk.Button(root, text="Run Selected", command=on_run).pack(pady=5)
root.mainloop()
#INFINITE LOOP WAITING ON GUI INPUT

#WE ARE TESTING A ROBOT, SO WE NEED TO DISCARD ALL INPUTS BETWEEN GUI INPUT AND OUTPUT FROM FUNCTION
