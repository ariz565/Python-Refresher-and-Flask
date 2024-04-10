import unittest
from app import app, db, ToDoModel

class TestToDoListAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup method to create a test database and some initial data
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
        cls.app = app.test_client()
        db.create_all()
        # Insert some initial data
        todo1 = ToDoModel(task="Task 1", summary="Summary 1", status="pending")
        todo2 = ToDoModel(task="Task 2", summary="Summary 2", status="completed")
        db.session.add(todo1)
        db.session.add(todo2)
        db.session.commit()

    @classmethod
    def tearDownClass(cls):
        # Teardown method to remove the test database
        db.drop_all()

    def test_get_all_todos(self):
        response = self.app.get('/todos')
        self.assertEqual(response.status_code, 200)
        todos = response.json
        self.assertEqual(len(todos), 2)  # Assuming you've inserted two todos in setUpClass

    def test_get_todo_by_id(self):
        response = self.app.get('/todos/1')
        self.assertEqual(response.status_code, 200)
        todo = response.json
        self.assertEqual(todo['id'], 1)
        self.assertEqual(todo['task'], 'Task 1')

    def test_get_nonexistent_todo_by_id(self):
        response = self.app.get('/todos/100')
        self.assertEqual(response.status_code, 404)
        error_message = response.json['message']
        self.assertEqual(error_message, 'Could not find task with that id')

    def test_create_todo(self):
        todo_data = {'task': 'New Task', 'summary': 'New Summary', 'status': 'pending'}
        response = self.app.post('/todos', json=todo_data)
        self.assertEqual(response.status_code, 201)
        new_todo = response.json
        self.assertIn('id', new_todo)
        self.assertEqual(new_todo['task'], 'New Task')

    def test_update_todo(self):
        todo_data = {'task': 'Updated Task', 'summary': 'Updated Summary', 'status': 'completed'}
        response = self.app.put('/todos/1', json=todo_data)
        self.assertEqual(response.status_code, 200)
        updated_todo = response.json
        self.assertEqual(updated_todo['task'], 'Updated Task')

    def test_delete_todo(self):
        response = self.app.delete('/todos/2')
        self.assertEqual(response.status_code, 204)

if __name__ == '__main__':
    unittest.main()
