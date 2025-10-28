"""
Auto-open the graph viewer in your default browser
"""
import webbrowser
import os
import sys

def main():
    html_file = 'view_graphs.html'
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found!")
        print("Please run: python data_science_analysis.py first")
        return
    
    # Get the absolute path
    abs_path = os.path.abspath(html_file)
    file_url = f'file:///{abs_path.replace(os.sep, "/")}'
    
    print("Opening graph viewer...")
    print(f"File: {abs_path}")
    
    # Open in browser
    webbrowser.open(file_url)
    print("\n✓ Graph viewer opened in your browser!")
    print("\nIf the graphs don't load, make sure you have:")
    print("  1. Generated the PNG files by running: python quick_demo.py")
    print("  2. Then run: python data_science_analysis.py")

if __name__ == '__main__':
    main()


