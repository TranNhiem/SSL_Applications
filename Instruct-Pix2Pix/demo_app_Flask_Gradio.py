from flask import Flask, render_template, request
import gradio as gr

app = Flask(__name__)

# Define your function to be used by Gradio interface
def greet(name):
    return f"Hello {name}!"

# Define Gradio interface
iface = gr.Interface(
    greet, 
    "text", 
    "text", 
    examples=[["Alice"], ["Bob"], ["Charlie"]],
    title="Greeting App",
    description="Enter a name to receive a greeting."
)

# Define Flask route to render Gradio interface
@app.route("/", methods=["GET", "POST"])
def gradio_interface():
    if request.method == "POST":
        return iface.process(request)
    return render_template("interface.html", iface=iface)

if __name__ == "__main__":
    app.run(debug=True)
