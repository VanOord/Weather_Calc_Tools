
import subprocess

def install_package(package):
    try:
        subprocess.check_call(['pip', 'install', '--upgrade', package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"Error installing {package}")

def main():

    # Install or update packages
    packages = [
        'pandas',
        'numpy',
        'calendar',
        'PySimpleGUI'
    ]

    for package in packages:
        install_package(package)

if __name__ == '__main__':
    main()
