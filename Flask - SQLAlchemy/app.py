from flask import Flask
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Initialize the Flask-Restful API
api = Api(app)

# Configure the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.sqlite3'

# Initialize the SQLAlchemy object with the Flask app
db = SQLAlchemy(app)

# Define your SQLAlchemy model
class ToDoModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task = db.Column(db.String(200))
    summary = db.Column(db.String(500))
    status = db.Column(db.String(200))

# Create the database tables based on the defined models
# with app.app_context():
#     db.create_all()


# todos = {
#     1:{"task":"prepare for the interview","summary":"prepare for the interview","status":"done"},
#     2:{"task":"Flask","summary":"prepare for the interview","status":"done"},
#     3:{"task":"AWS","summary":"prepare for the interview","status":"done"},
# }



task_post_args = reqparse.RequestParser()
task_post_args.add_argument("task", type=str, help="Task is required", required=True)
task_post_args.add_argument("summary", type=str, help="Summary is required", required=True)
task_post_args.add_argument("status", type=str, help="Status is required", required=True)

task_put_args = reqparse.RequestParser()
task_put_args.add_argument('task', type=str)
task_put_args.add_argument('summary', type=str)
task_put_args.add_argument('status', type=str)

resource_fields = {
    'id': fields.Integer,
    'task': fields.String,
    'summary': fields.String,
    'status': fields.String,
}


class ToDoList(Resource):
    def get(self):
        tasks = ToDoModel.query.all()
        todos = {}
        for task in tasks:
            todos[task.id] = {"task":task.task, "summary": task.summary, "status": task.status}
        return todos
    
class ToDo(Resource):
    @marshal_with(resource_fields)
    def get(self,todo_id):
        task = ToDoModel.query.filter_by(id=todo_id).first()
        if not task:
            abort(404, message="Could not find task with that id")
        return task
        # return todos[todo_id]
    
    @marshal_with(resource_fields)
    def post(seld, todo_id):
        args = task_post_args.parse_args()
        
        task = ToDoModel.query.filter_by(id=todo_id).first()
        if task:
            abort(409, message="Task id taken")
            
        todo = ToDoModel(id=todo_id, task=args['task'], summary=args['summary'], status=args['status'])
        db.session.add(todo)
        db.session.commit()
        return todo, 201
        # if todo_id in todos:
        #     abort(409, message="Todo id already taken")
        # todos[todo_id] = {"task":args["task"], "summary": args["summary"], "status": args["status"]}
        # return todos[todo_id], 201  
    
    @marshal_with(resource_fields)
    def put(self,todo_id):
        args = task_put_args.parse_args()
        task = ToDoModel.query.filter_by(id=todo_id).first()
        if not task:
            abort(404, message="Task doesnt exist, cannot update")
        if args["task"]:
            task.task = args["task"]
        if args["summary"]:
            task.summary = args["summary"]
        if args["status"]:
            task.status = args["status"]
        db.session.commit()
        return task
        # if todo_id not in todos:
        #     abort(404, message="Task doesnt exist")
        # if args["task"]:
        #     todos[todo_id]["task"] = args["task"]
        # if args["summary"]:
        #     todos[todo_id]["summary"] = args["summary"]
        # if args["status"]:
        #     todos[todo_id]["status"] = args["status"]
        # return todos[todo_id]
        
    def delete(self,todo_id):
        task= ToDoModel.query.filter_by(id=todo_id).delete()
        db.session.commit()
        return 'Todo Deleted', 204 
        # del todos[todo_id]
        # return todos
        
    
api.add_resource(ToDo, '/todos/<int:todo_id>')
api.add_resource(ToDoList, '/todos')






# class HelloWorld(Resource):
#     def get(self):
#         return render_template('index.html')
    

# @app.route('/users')
# def get_users():
#     users = [
#         {"id": 1, "name": "Alice"},
#         {"id": 2, "name": "Bob"}
#     ]
#     return jsonify(users)

    
# api.add_resource(HelloWorld, '/helloworld')

if __name__ == '__main__':
    app.run(debug=True)
    