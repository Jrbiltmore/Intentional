import os

def create_module(module_name):
    # Create the module directory
    module_dir = module_name.lower()
    os.makedirs(module_dir, exist_ok=True)

    # Create the module file
    module_file = f"{module_dir}/{module_name}.py"
    with open(module_file, "w") as f:
        f.write("# Intentional content goes here\n")
        f.write(f"def hello():\n")
        f.write(f"    print('Hello from {module_name} module!')\n")

    print(f"Module {module_name} created successfully in {module_dir} directory.")

if __name__ == '__main__':
    module_name = input("Enter the name of the module: ")
    create_module(module_name)
