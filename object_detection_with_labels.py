import argparse
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageFont, ImageDraw

# Function to draw a rectangle with width > 1
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='Path of the detection model.', required=True)
  parser.add_argument(
      '--label', help='Path of the labels file.')
  parser.add_argument(
      '--input', help='File path of the input image.', required=True)
  parser.add_argument(
      '--output', help='File path of the output image.')
  args = parser.parse_args()

  if not args.output:
    output_name = 'object_detection_result.jpg'
  else:
    output_name = args.output

  # Initialize engine.
  engine = DetectionEngine(args.model)
  labels = ReadLabelFile(args.label) if args.label else None

  # Open image.
  img = Image.open(args.input)
  draw = ImageDraw.Draw(img)
  helvetica=ImageFont.truetype("./Helvetica.ttf", size=72)
  #defaultfont = ImageFont.load_default()
  
  # Run inference.
  ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,
                               relative_coord=False, top_k=10)

  # Display result.
  if ans:
    print("\n")
    for obj in ans:
      if obj.score > 0.75:
         if labels:
           print(labels[obj.label_id], 'score = ', obj.score)
         else:
           print ('score = ', obj.score)
         box = obj.bounding_box.flatten().tolist()
         #print ('box = ', box)
         # Draw a rectangle.
         draw_rectangle(draw, box, 'red', width=5)
         if labels:
           draw.text((box[0] + 20, box[1] + 20), labels[obj.label_id], fill='red', font=helvetica)
    img.save(output_name)
    print ('Saved to ', output_name)
  else:
    print ('No object detected!')

if __name__ == '__main__':
  main()