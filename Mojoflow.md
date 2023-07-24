import json
import time
import sqlite3
import threading
import logging
import pickle

class MojoFlow:
    """
    Represents a sophisticated MojoFlow workflow.

    Attributes:
        workflow_name (str): The name of the MojoFlow workflow.
        workflow_steps (list): A list of workflow steps.
        step_dependencies (dict): A dictionary representing step dependencies.
    """

    def __init__(self, workflow_name, db_path=None):
        """
        Initialize a MojoFlow workflow.

        Args:
            workflow_name (str): The name of the MojoFlow workflow.
            db_path (str): Optional path to the SQLite database for workflow data storage.
        """
        self.workflow_name = workflow_name
        self.workflow_steps = []
        self.step_dependencies = {}
        self.db_path = db_path
        if db_path:
            self.create_db_table()

    def create_db_table(self):
        """
        Create a database table to store workflow details.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS workflow_details
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      workflow_name TEXT,
                      step_name TEXT,
                      execution_time REAL)''')
        conn.commit()
        conn.close()

    def add_step(self, step_name, step_func=None, execution_time=0, before_execute=None, after_execute=None,
                 dependencies=None, **step_attributes):
        """
        Add a workflow step to the MojoFlow workflow.

        Args:
            step_name (str): The name of the step to add.
            step_func (callable): A custom function for executing the step (optional).
            execution_time (float): Custom execution time for the step (optional).
            before_execute (callable): A custom function to execute before the step (optional).
            after_execute (callable): A custom function to execute after the step (optional).
            dependencies (list): A list of step names that this step depends on (optional).
            **step_attributes: Additional attributes for the step (optional).
        """
        step = {'name': step_name, 'func': step_func, 'execution_time': execution_time,
                'before_execute': before_execute, 'after_execute': after_execute, 'attributes': step_attributes}
        self.workflow_steps.append(step)

        if dependencies:
            self.step_dependencies[step_name] = dependencies

    def remove_step(self, step_name):
        """
        Remove a workflow step from the MojoFlow workflow.

        Args:
            step_name (str): The name of the step to remove.
        """
        try:
            self.workflow_steps = [step for step in self.workflow_steps if step['name'] != step_name]
            self.step_dependencies.pop(step_name, None)
            for step_name, dependencies in self.step_dependencies.items():
                if step_name in dependencies:
                    dependencies.remove(step_name)
        except ValueError:
            raise ValueError(f"Step '{step_name}' not found in the workflow.")

    def clear_steps(self):
        """
        Clear all workflow steps from the MojoFlow workflow.
        """
        self.workflow_steps = []
        self.step_dependencies = {}

    def display_workflow(self):
        """
        Display the MojoFlow workflow with its steps and attributes.
        """
        print(f"MojoFlow Workflow: {self.workflow_name}")
        if self.workflow_steps:
            print("Steps:")
            for idx, step in enumerate(self.workflow_steps, start=1):
                print(f"{idx}. {step['name']}")
                if step['execution_time'] > 0:
                    print(f"   - Execution Time: {step['execution_time']} seconds")
                if step['attributes']:
                    print(f"   - Attributes: {step['attributes']}")
            if self.step_dependencies:
                print("Dependencies:")
                for step_name, dependencies in self.step_dependencies.items():
                    print(f"{step_name} depends on: {', '.join(dependencies)}")
        else:
            print("No steps added to the workflow yet.")

    def execute_step(self, step):
        """
        Execute a single step in the MojoFlow workflow.

        Args:
            step (dict): The step dictionary containing step details.

        Returns:
            bool: True if the step is successfully executed, False otherwise.
        """
        try:
            if step['before_execute']:
                step['before_execute'](**step['attributes'])
            if step['func']:
                start_time = time.time()
                step['func'](**step['attributes'])
                end_time = time.time()
                step['execution_time'] = round(end_time - start_time, 4)
            if step['after_execute']:
                step['after_execute'](**step['attributes'])
            return True
        except Exception as e:
            logging.error(f"Error during step execution: {str(e)}")
            return False

    def execute(self, start_step=0, end_step=None, report_progress=True, parallel=False):
        """
        Execute the MojoFlow workflow.

        Args:
            start_step (int): The index of the step to start execution (default: 0).
            end_step (int): The index of the step to end execution (inclusive).

        Note:
            If `end_step` is not specified, execution will continue until the last step.
            Set `report_progress` to False to disable progress reporting during execution.
            Set `parallel` to True to execute steps in parallel using multiple threads.

        Returns:
            bool: True if the entire workflow is executed successfully, False otherwise.
        """
        if not self.workflow_steps:
            logging.warning("The workflow is empty. Add steps to execute.")
            return False

        if end_step is None:
            end_step = len(self.workflow_steps)

        if start_step >= len(self.workflow_steps) or end_step < start_step:
            raise ValueError("Invalid start or end step index.")

        logging.info(f"Executing MojoFlow workflow '{self.workflow_name}'...")
        if parallel:
            threads = []
            for idx in range(start_step, min(end_step, len(self.workflow_steps))):
                step = self.workflow_steps[idx]
                logging.info(f"Step {idx + 1}: {step['name']} (in parallel)")
                thread = threading.Thread(target=self.execute_step, args=(step,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for idx in range(start_step, min(end_step, len(self.workflow_steps))):
                step = self.workflow_steps[idx]
                logging.info(f"Step {idx + 1}: {step['name']}")
                if report_progress:
                    logging.info("Step in progress...")
                if not self.execute_step(step):
                    logging.error(f"Step {idx + 1} failed to execute.")
                    return False

        if self.db_path:
            self.save_workflow_details()

        logging.info("MojoFlow workflow execution completed successfully.")
        return True

    def save_workflow_details(self):
        """
        Save the workflow details to the SQLite database.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for step in self.workflow_steps:
            c.execute("INSERT INTO workflow_details (workflow_name, step_name, execution_time) VALUES (?, ?, ?)",
                      (self.workflow_name, step['name'], step['execution_time']))
        conn.commit()
        conn.close()

    def serialize_to_json(self, output_path):
        """
        Serialize the MojoFlow object to a JSON file.

        Args:
            output_path (str): The path to save the JSON file.
        """
        try:
            with open(output_path, 'w') as json_file:
                data = {
                    'workflow_name': self.workflow_name,
                    'workflow_steps': self.workflow_steps,
                    'step_dependencies': self.step_dependencies
                }
                json.dump(data, json_file, indent=4)
            logging.info(f"MojoFlow serialized to '{output_path}'.")
        except Exception as e:
            logging.error(f"Error while serializing MojoFlow: {str(e)}")

    @classmethod
    def deserialize_from_json(cls, file_path):
        """
        Deserialize a MojoFlow object from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            MojoFlow: The deserialized MojoFlow object.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                mojoflow = cls(data['workflow_name'])
                mojoflow.workflow_steps = data['workflow_steps']
                mojoflow.step_dependencies = data['step_dependencies']
            logging.info(f"MojoFlow deserialized from '{file_path}'.")
            return mojoflow
        except Exception as e:
            logging.error(f"Error while deserializing MojoFlow: {str(e)}")
            return None

    def serialize_to_pickle(self, output_path):
        """
        Serialize the MojoFlow object to a pickle file.

        Args:
            output_path (str): The path to save the pickle file.
        """
        try:
            with open(output_path, 'wb') as pickle_file:
                pickle.dump(self, pickle_file)
            logging.info(f"MojoFlow serialized to '{output_path}'.")
        except Exception as e:
            logging.error(f"Error while serializing MojoFlow: {str(e)}")

    @classmethod
    def deserialize_from_pickle(cls, file_path):
        """
        Deserialize a MojoFlow object from a pickle file.

        Args:
            file_path (str): The path to the pickle file.

        Returns:
            MojoFlow: The deserialized MojoFlow object.
        """
        try:
            with open(file_path, 'rb') as pickle_file:
                mojoflow = pickle.load(pickle_file)
            logging.info(f"MojoFlow deserialized from '{file_path}'.")
            return mojoflow
        except Exception as e:
            logging.error(f"Error while deserializing MojoFlow: {str(e)}")
            return None

# Example usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    workflow = MojoFlow("MyWorkflow", db_path="workflow_db.sqlite")
    workflow.add_step("Step 1", step_func=lambda: print("Executing Step 1..."))
    workflow.add_step("Step 2", step_func=lambda x: print(f"Executing Step 2 with argument: {x}"), arg_value=42,
                      dependencies=["Step 1"])

    workflow.display_workflow()

    # Execute the entire workflow in parallel
    if workflow.execute(parallel=True):
        logging.info("Workflow executed successfully.")

    # Save the MojoFlow object to a JSON file
    workflow.serialize_to_json("myworkflow.json")

    # Deserialize the MojoFlow object from the JSON file
    loaded_workflow = MojoFlow.deserialize_from_json("myworkflow.json")
    if loaded_workflow:
        loaded_workflow.display_workflow()

        # Execute only steps 1 and 2 (indices 0 and 1) sequentially
        if loaded_workflow.execute(start_step=0, end_step=1, parallel=False):
            logging.info("Partial workflow executed successfully.")

        # Save the MojoFlow object to a pickle file
        loaded_workflow.serialize_to_pickle("myworkflow.pkl")

        # Deserialize the MojoFlow object from the pickle file
        deserialized_workflow = MojoFlow.deserialize_from_pickle("myworkflow.pkl")
        if deserialized_workflow:
            deserialized_workflow.display_workflow()
            # Visualize the workflow as a graph
            visualize_workflow_graph(deserialized_workflow)
