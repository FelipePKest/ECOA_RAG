# from model import Database
from model import DocumentDatabase
from view import *
from controller import ChatController


if __name__ == "__main__":
    # Initialize MVC components
    model = DocumentDatabase(file_path="data/Dominios sobre impacto socioambiental positivo")
    view = ChatView(file_path="data/Dominios sobre impacto socioambiental positivo")
    controller = ChatController(model, view)

    # Run the chat interface
    controller.run()
    # view.display()