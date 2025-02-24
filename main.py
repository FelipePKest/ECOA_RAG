# from model import Database
from model import DocumentDatabase
from view import *
from controller import ChatController


if __name__ == "__main__":
    # Initialize MVC components
    model = DocumentDatabase(file_path="data/pdfs")
    view = ChatView()
    controller = ChatController(model, view)

    # Run the chat interface
    controller.run()
    # view.display()