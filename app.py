import gradio as gr
from ultralytics import YOLO
import cv2

examples=[["photo/a.jpg","Image1"],["photo/b.jpg","Image2"],
          ["photo/c.jpg","Image3"],["photo/d.jpg","Image4"],
          ["photo/e.jpg","Image5"],["photo/f.jpg","Image6"],
          ["photo/g.jpg","Image7"],["photo/h.jpg","Image8"]]


def detect_objects_on_image(image_path):
    image = cv2.imread(image_path)
    model = YOLO("best.pt")
    results = model.predict(image_path)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
            
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        
        cv2.putText(image,result.names[class_id],  (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
demo = gr.Interface(
    fn=detect_objects_on_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Yolov8 Custom Object Detection",
    examples=examples,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()