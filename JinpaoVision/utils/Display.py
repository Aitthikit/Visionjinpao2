import cv2

class DISPLAY:
    def __init__(self) -> None:
        self.color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
        }

    def show_detect(self, pred_list, frame):
        
        for row in pred_list:
                x1, y1, x2, y2, conf, class_label = row
                if conf >= 0.5:
                    # Convert float to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw the bounding box
                    # BoxType = BoxClass[int(class_label)].split('_')
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_map[class_label.split('_')[0]], 2)

                    # Display class label
                    label = f"Class: {class_label} : {int(conf*100)}"

                    # Put text on the image
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                