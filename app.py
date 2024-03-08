from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class UppercaseText(Resource):

    def get(self):
        text = request.args.get('text')

        return jsonify({"text": text.upper()})

api.add_resource(UppercaseText, "/uppercase")

# if __name__ == "__main__":
#     app.run(debug=True)