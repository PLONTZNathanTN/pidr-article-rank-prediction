import os
import time
import pyautogui
import webbrowser


def download_arxiv_pdf(arxiv_url, file_number, index):
    # Define the save folder relative to the script location
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(project_dir, "data", "pdf")
    print(project_dir)

    # Extract the article ID from the URL
    arxiv_id = arxiv_url.split("/")[-1]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    file_name = f"{arxiv_id}.pdf"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Check if the file already exists
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f"File already exists {index}: {save_path}")
        return 2

    # Open the PDF link in the default web browser
    webbrowser.open(pdf_url)
    
    # Wait for the page to load
    time.sleep(2.5)

    # Simulate Ctrl+S to open the save dialog
    pyautogui.keyDown('ctrl')
    pyautogui.press('s')
    pyautogui.keyUp('ctrl')

    # Wait for the save dialog to appear
    time.sleep(1)

    # Simulate pressing Enter to save with default name and location
    pyautogui.press('enter')

    # Check if the file was downloaded
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        if file_size < 5 * 1024:  # Less than 5 KB → likely a CAPTCHA page
            os.remove(save_path)
            print(f"File {arxiv_id} could not be downloaded (CAPTCHA detected). Please solve it manually.")
            pyautogui.hotkey('ctrl', 'w')  # Close the tab
            return 0
        else:
            print(f"PDF {file_number} downloaded successfully: {save_path}")
            pyautogui.hotkey('ctrl', 'w')  # Close the tab
            return 1
    else:
        print("Error during download. Please try manually.")
        pyautogui.hotkey('ctrl', 'w')
        return 0
