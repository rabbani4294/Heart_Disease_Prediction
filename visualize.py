import customtkinter as ctk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ================= LOAD CSV =================
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ================= GUI SETUP =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Heart Disease Prediction Dashboard")
app.geometry("1000x700")

# ================= MAIN FRAMES =================
left_frame = ctk.CTkFrame(app, width=500, corner_radius=10)
left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

right_frame = ctk.CTkFrame(app, width=500, corner_radius=10)
right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

# ================= TITLE =================
ctk.CTkLabel(
    left_frame,
    text="Heart Disease Prediction",
    font=("Arial", 24, "bold")
).pack(pady=20)

# ================= FORM =================
form = ctk.CTkFrame(left_frame, corner_radius=10)
form.pack(padx=20, pady=10, fill="both", expand=True)

columns = X.columns.tolist()
entries = {}

for col in columns:
    row = ctk.CTkFrame(form)
    row.pack(fill="x", pady=5, padx=10)

    ctk.CTkLabel(row, text=col, width=150, anchor="w").pack(side="left", padx=10)
    ent = ctk.CTkEntry(row, width=200)
    ent.pack(side="right", padx=10)

    entries[col] = ent

# ================= RESULT LABEL =================
result_label = ctk.CTkLabel(
    left_frame,
    text="Result: ---",
    font=("Arial", 20, "bold")
)
result_label.pack(pady=20)

# ================= PREDICT FUNCTION =================
def predict():
    try:
        values = [float(entries[col].get()) for col in columns]
        values_scaled = scaler.transform([values])
        pred = model.predict(values_scaled)[0]

        if pred == 1:
            result_label.configure(
                text="Result: Heart Disease Detected",
                text_color="red"
            )
        else:
            result_label.configure(
                text="Result: No Heart Disease",
                text_color="green"
            )
    except:
        result_label.configure(
            text="Result: Invalid Input",
            text_color="orange"
        )

# ================= GRAPH FUNCTION =================
def open_graph_window():
    graph_win = ctk.CTkToplevel(app)
    graph_win.title("Heart Disease Visualization")
    graph_win.geometry("1000x600")

    container = ctk.CTkFrame(graph_win)
    container.pack(fill="both", expand=True, padx=20, pady=20)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Bar Chart
    df["target"].value_counts().plot(
        kind="bar",
        ax=axs[0],
        color=["green", "red"],
        title="Heart Disease Distribution"
    )
    axs[0].set_xlabel("Target")
    axs[0].set_ylabel("Count")

    # Histogram
    axs[1].hist(df["age"], bins=20, color="skyblue")
    axs[1].set_title("Age Distribution")
    axs[1].set_xlabel("Age")
    axs[1].set_ylabel("Frequency")

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ================= BUTTONS =================
button_frame = ctk.CTkFrame(right_frame)
button_frame.pack(pady=50)

ctk.CTkButton(
    button_frame,
    text="Predict Heart Disease",
    width=250,
    height=45,
    font=("Arial", 16, "bold"),
    command=predict
).pack(pady=(0, 20))

ctk.CTkButton(
    button_frame,
    text="Graphical Visualization",
    width=250,
    height=45,
    font=("Arial", 16, "bold"),
    command=open_graph_window
).pack()

# ================= START APP =================
app.mainloop()
