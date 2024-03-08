from flask import Flask, jsonify, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return jsonify({"message": "Hello Universe!"})

api.add_resource(HelloWorld, "/")

if __name__ == "__main__":
    app.run(debug=True)