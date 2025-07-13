#!/usr/bin/env python3

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from paddleocr import PaddleOCR
from translate import Translator as TranslatorEngine
from pathlib import Path
import json
import argparse
import sys
import os


class Input:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Automatically recognize Japanese text from images and translate it to English",
            epilog="For more information, visit https://github.com/manfanocr/ManfanOCR",
        )
        self.parser.add_argument(
            nargs='*',
            default=[],
            dest='images',
            metavar='FILE',
        )
        self.parser.add_argument(
            '-s', '--skip-ocr',
            dest='skip_ocr',
            action='store_true',
            help="skip OCR for previously processed files",
        )
        self.parser.add_argument(
            '-d', '--debug',
            dest='debug',
            action='store_true',
            help="enable debug mode",
        )
    def run(self):
        self.arguments = self.parser.parse_args()
    def get_images(self):
        return self.arguments.images
    def get_is_debug(self):
        return self.arguments.debug
    def get_skip_ocr(self):
        return self.arguments.skip_ocr


class Reader:
    def __init__(self, images, skip_ocr):
        self.images = images
        self.skip_ocr = skip_ocr
        self.processed = []
        self.engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    def run(self):
        for image in self.images:
            if not os.path.isfile(image):
                print(f"File not found: '{image}'")
                continue
            if not Reader.is_image_file(image):
                print(f"Invalid file format: '{image}'")
                continue
            print(f"Reading image: '{image}'")
            if self.skip_ocr and os.path.isfile("output/" + Path(image).stem + "_res.json"):
                print(f"Skipping OCR for '{image}'")
                self.processed.append(image)
                continue
            result = self.engine.predict(input=image)
            for res in result:
                res.save_to_json("output")
            self.processed.append(image)
    def get_images(self):
        return self.processed
    def is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Box:
    def __init__(self, text, box, score):
        self.x = box[0]
        self.w = box[2] - box[0]
        self.y = box[1]
        self.h = box[3] - box[1]
        self.text = text
        self.score = score
    def overlaps(self, other):
        # Fixed overlap detection using AABB logic
        return (self.x <= other.x + other.w and
                self.x + self.w >= other.x and
                self.y <= other.y + other.h and
                self.y + self.h >= other.y)
    def debug_print(self):
        print(f"Box:")
        print(f"  Text: {self.text}")
        print(f"  Score: {self.score}")
        print(f"  Coordinates: {self.x}, {self.y}, {self.w}, {self.h}")


class BoxGroup:
    def __init__(self, box):
        self.boxes = [box,]
        self.full_text = ""
        # init -> X1, Y1, X2, Y2; finish -> X, Y, W, H
        self.full_box = [box.x, box.y, box.x + box.w, box.y + box.h,]
        self.translation = ""
    def find_next_neighbor(self, box, all_boxes, processed_boxes):
        for other_box in all_boxes:
            if other_box.overlaps(box) and other_box not in processed_boxes:
                self.boxes.append(other_box)
                processed_boxes.append(other_box)
                self.find_next_neighbor(other_box, all_boxes, processed_boxes)
    def debug_print(self):
        print("BoxGroup:")
        print(f"  Full Text: {self.full_text}")
        print(f"  Full Box: {self.full_box}")
        for box in self.boxes:
            box.debug_print()
    def finish_init(self):
        for index in range(len(self.boxes) - 1, -1, -1):
            self.full_text += self.boxes[index].text.strip()
        for box in self.boxes:
            self.full_box[0] = min(box.x, self.full_box[0])
            self.full_box[1] = min(box.y, self.full_box[1])
            self.full_box[2] = max(box.x + box.w, self.full_box[2])
            self.full_box[3] = max(box.y + box.h, self.full_box[3])
        self.full_box[2] -= self.full_box[0]
        self.full_box[3] -= self.full_box[1]


class Page:
    SCORE_THRESHOLD = 0.75
    def __init__(self, image):
        self.image = image
        self.data = {}
        self.boxes = []
        self.groups = []
    def run(self):
        self.load_data()
        self.load_boxes()
        self.make_groups()
    def load_data(self):
        json_path = "output/" + Path(self.image).stem + '_res.json'
        with open(json_path, 'r') as file:
            self.data = json.load(file)
    def load_boxes(self):
        for index in range(len(self.data["rec_texts"])):
            if self.data["rec_scores"][index] >= Page.SCORE_THRESHOLD:
                new_box = Box(
                    self.data["rec_texts"][index],
                    self.data["rec_boxes"][index],
                    self.data["rec_scores"][index]
                )
                self.boxes.append(new_box)
    def make_groups(self):
        processed_boxes = []
        for box in self.boxes:
            if box not in processed_boxes:
                new_group = BoxGroup(box)
                self.groups.append(new_group)
                processed_boxes.append(box)
                new_group.find_next_neighbor(box, self.boxes, processed_boxes)
                new_group.finish_init()
    def debug_print(self):
        print("Number of groups:", len(self.groups))
        for group in self.groups:
            group.debug_print()


class Pager:
    def __init__(self, images, is_debug):
        self.images = images
        self.is_debug = is_debug
        self.pages = []
    def run(self):
        for image in self.images:
            page = Page(image)
            page.run()
            self.pages.append(page)
        self.debug_print()
    def get_pages(self):
        return self.pages
    def debug_print(self):
        if not self.is_debug:
            return
        for page in self.pages:
            print(f"Number of pages: {len(self.pages)}")
            page.debug_print()


class Translator:
    def __init__(self, pages):
        self.pages = pages
        self.engine = TranslatorEngine(from_lang="ja", to_lang="en")
    def run(self):
        for page in self.pages:
            print(f"Translating image: '{page.image}'")
            for group in page.groups:
                group.translation = self.engine.translate(group.full_text)
    def get_pages(self):
        return self.pages


class Drawer:
    def __init__(self, pages, is_debug):
        self.pages = pages
        self.is_debug = is_debug
        self.font = ImageFont.truetype("/home/farkasau/arial.ttf", 24)
    def run(self):
        for page in self.pages:
            print(f"Drawing image: '{page.image}'")
            image = Image.open(page.image)
            draw = ImageDraw.Draw(image)
            for group in page.groups:
                Drawer.fill_old_text(group, draw)
                self.debug_draw(group, draw)
                self.draw_translation(group, draw)
                image.save("output/" + Path(page.image).stem + ".jpg", "JPEG")
        print("Done!")
    def fill_old_text(group, draw):
        for box in group.boxes:
            draw.rectangle(((box.x, box.y,), (box.x + box.w, box.y + box.h,),), fill="white")
    def debug_draw(self, group, draw):
        if not self.is_debug:
            return
        box = group.full_box
        draw.rectangle(((box[0], box[1],), (box[0] + box[2], box[1] + box[3],),), outline="red")
    def get_wrapped_text(text, font, width):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if font.getlength(line) <= width:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)
    def draw_translation(self, group, draw):
        wrapped_text = Drawer.get_wrapped_text(group.translation, self.font, group.full_box[2])
        draw.multiline_text((group.full_box[0], group.full_box[1],), wrapped_text, font=self.font, fill="black")


if __name__ == "__main__":
    parser = Input()
    parser.run()
    reader = Reader(parser.get_images(), parser.get_skip_ocr())
    reader.run()
    pager = Pager(reader.get_images(), parser.get_is_debug())
    pager.run()
    translator = Translator(pager.get_pages())
    translator.run()
    drawer = Drawer(translator.get_pages(), parser.get_is_debug())
    drawer.run()
