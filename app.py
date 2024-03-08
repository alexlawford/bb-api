from flask import Flask 

# from flask import Flask, jsonify, request
# from flask_restful import Api, Resource

# app = Flask(__name__)
# api = Api(app)

# class UppercaseText(Resource):

#     def get(self):
#         text = request.args.get('text')

#         return jsonify({"text": text.upper()})

# api.add_resource(UppercaseText, "/uppercase")

# # if __name__ == "__main__":
# #     app.run(debug=True)

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Deployed!</h1><style>body { display: flex; align-items: center; justify-content: center; height: 100vh; }</style>'