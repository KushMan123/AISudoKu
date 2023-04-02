import cv2
import keyboard
import numpy as np
from utilis import load_model, predict_digit, convert_to_grid, convert_to_list, get_corners, sort_corners, show_image, format_grid
from solve_sudoku import find_solution
from sudoku import Sudoku
from model.pytorch_digit_classifier import Model
import argparse

print("[INFO] Setting device")
device="cpu"

print("[INFO] Loading model")
MODEL_FILE_NAME="output/ocr_model.pt"
model=load_model(MODEL_FILE_NAME ,device)

def process_image(img, debug=False):
    #Image Preprocessong
    img_height, img_width, _= img.shape
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    img_out = img.copy()

    if debug:
        show_image("Image Thresholding", thresh)

    # Extracting the biggest grid with Contours
    # Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity.

    contours,_=cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grille=None
    max_area=0
    solution_found=False
    for c in contours:
        area=cv2.contourArea(c) #Calculate the area of the contours
        if area>50000:
            peri=cv2.arcLength(c,True) #Calaculate the perimeter of the closed surface
            polygon=cv2.approxPolyDP(c, 0.01*peri, True) # approzimate the given shape with a simpler image consiting of fewer points while still preserving the original shape
            if area>max_area and len(polygon) == 4:
                contour_grille=polygon
                max_area = area
        
    if contour_grille is not None:
        print("[INFO] Drawing sudoku contour using lines and points")
            # Draw sudoku contour using lines and points
        cv2.line(img_out, polygon[0][0], polygon[1][0], (0, 200, 0), thickness=3)
        cv2.line(img_out, polygon[1][0], polygon[2][0], (0, 200, 0), thickness=3)
        cv2.line(img_out, polygon[2][0], polygon[3][0], (0, 200, 0), thickness=3)
        cv2.line(img_out, polygon[0][0], polygon[3][0], (0, 200, 0), thickness=3)
        cv2.drawContours(img_out, polygon, -1, (0, 255, 255), 10)

        if debug:
            show_image("Contour Detected", img_out)
        # corners= get_corners(contour_grille)
        # # Sorting the corner points tl tr bl br
        points_1= np.float32(sort_corners(polygon))
        (tl,tr,br,bl)=points_1
        #Determining the width and height of the sudoku
        width_A=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
        width_B=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))

        height_A=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
        height_B=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))

        max_width=max(int(width_A), int(width_B))
        max_height=max(int(height_A), int(height_B))

        # Destination point to map the board
        board_size=max(max_width,max_height)
        dst=np.array([
            [0,0],[board_size,0],[board_size,board_size],[0,board_size]
        ], dtype="float32")

        # Calaculate the perspective transform matrix and the warp the board to the screen
        t_matrix=cv2.getPerspectiveTransform(points_1,dst)
        _, t_matrix_inv= cv2.invert(t_matrix)
        # print("[INFO] Perspective Transform Matrix")
        # print(t_matrix)
        warp= cv2.warpPerspective(img_out, t_matrix,(board_size,board_size))
        
        if debug:
            show_image("Extracted Image", warp)

        print("[INFO] Image Extracted")

        #Extracted Image Processing
        p_window=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
        p_window=cv2.GaussianBlur(p_window,(5,5),0)
        p_window=cv2.adaptiveThreshold(p_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        warp_bw=cv2.bitwise_not(p_window)
        _,warp_bw= cv2.threshold(warp_bw,200,255,cv2.THRESH_BINARY)

        if debug:
            show_image("P Window", p_window)
            show_image("P", warp_bw)

        #Extract indicate
        print("[INFO] Extracting Digit")
        images=[]
        cell_w, cell_h = board_size//9, board_size//9
        for i in range(9):
            for j in range(9):
                x_min= cell_w * i
                x_max= cell_w * (i+1)
                y_min= cell_h * j
                y_max= cell_h * (j+1)
                cx=(x_min+x_max)//2
                cy=(y_min+y_max)//2
                crop_image=warp_bw[y_min:y_max,x_min:x_max]
                cntrs, _= cv2.findContours(crop_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                for dc in cntrs:
                    img_x, img_y, img_w, img_h= cv2.boundingRect(dc)
                    if 0.2* cell_w < img_w < 0.8 * cell_w and 0.4* cell_h < img_h < 0.8*cell_h:
                        cv2.rectangle(warp,(x_min+img_x, y_min+img_y),(x_min+img_x+img_w, y_min+img_y+img_h),(0,255,0),1)
                        digit_img = crop_image[img_y:img_y+img_h, img_x: img_x+img_w]
                        images.append((i,j,cx,cy,digit_img))
                        
                        # if debug:
                        #     show_image("Extracted digit", warp)
                            #show_image("Extract digit",digit_img) 

        print("[INFO] Predicting Digits")
        results=predict_digit(model,images,32,device)
        print(results)
        board=[0]*(9*9)
        for(x,y,img_x,img_y,digit_img), result in zip(images, results):
            board[9*x+y]=result
            img_c=np.array([[[img_x, img_y]]],dtype=np.float32)
            orig=cv2.perspectiveTransform(img_c,t_matrix_inv)
            orig=orig.reshape(2,).astype(np.int32)
            cv2.putText(img_out,str(result),orig,cv2.FONT_HERSHEY_SIMPLEX, 1.05 , (200,0,0),2, cv2.LINE_AA, False)
        
        if debug:
            show_image("Image with predicted Text", img_out)

        #Finding Solution
        print("[INFO] Finding Solution")
        print(board)
        grid_board= convert_to_grid(board)
        predicted_board=format_grid(grid_board)
        print(predicted_board)
        solve_grid_board, solution_found = find_solution(grid_board)
        board_orig = list(board)
        solved_board = convert_to_list(solve_grid_board)
        if solution_found:
            print("[INFO] Solution Found")
            for x in range(9):
                for y in range(9):
                    if(board[9*x+y]==0):
                        pt_x, pt_y=x*cell_w+cell_w//2, y*cell_h+cell_h//2
                        pt_origin=cv2.perspectiveTransform(np.array([[[pt_x,pt_y]]],dtype=np.float32),t_matrix_inv).reshape(2,).astype(np.int32)
                        cv2.putText(img_out,str(solved_board[9*x+y]),pt_origin, cv2.FONT_HERSHEY_COMPLEX ,1,(0,0,200),2,cv2.LINE_AA, False)
        else:
            print("[INFO] Solution Not Found")
        
        cv2.putText(img_out, "Solution Found" if solution_found else "Solution Not Found", (10, img_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0 ), 2, cv2.LINE_AA, False)

        if debug:
            show_image("Solution Set", img_out)
    
    return solution_found, img_out,predicted_board


if __name__=="__main__":
    #ArgParser
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--image", required=True, help="images/1.png")
    ap.add_argument("-d","--debug", type=int, default=-1, help="debug")
    args=vars(ap.parse_args())

    print("[INFO] Setting device")
    device="cpu"

    print("[INFO] Loading model")
    MODEL_FILE_NAME="output/ocr_model.pt"
    model=load_model(MODEL_FILE_NAME ,device)

    #Reading the Image
    print("[INFO] Reading the Image")
    image=args["image"]
    img=cv2.imread(image)
    
    solution_found, frame= process_image(model, img, device)
    
    show_image("Solution", frame)

