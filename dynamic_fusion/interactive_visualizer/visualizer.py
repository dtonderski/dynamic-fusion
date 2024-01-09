from pathlib import Path
from typing import Any, Optional

import customtkinter as ctk
from customtkinter import filedialog
import numpy as np
from ruamel.yaml import YAML

from dynamic_fusion.interactive_visualizer.configuration import (
    VisualizerConfiguration,
)
from dynamic_fusion.interactive_visualizer.network_handler import NetworkHandler
from PIL import Image, ImageTk, ImageDraw

NO_DATA_TEXT = "No data folder selected!"
START_BIN_INDEX_TEMPLATE = "Start bin index: {val:.0f}"
END_BIN_INDEX_TEMPLATE = "End bin index: {val:.0f}"
GRID_LINES_PIXEL_SPACING = 20
TIMESTAMP_TEMPLATE = "Timestamp in bin: {val:.2f}"
GT_TEMPLATE = "Ground truth at: {val:.2f}"
PREDICTION_TEMPLATE = "Reconstruction at {val:.2f}"
IMAGE_SIZE = (350, 350)

ctk.set_appearance_mode("dark")


class Visualizer(ctk.CTk):  # type: ignore
    config: VisualizerConfiguration
    row: int = 0
    network_handler: NetworkHandler
    start_image: ImageTk.PhotoImage
    end_image: ImageTk.PhotoImage
    x_start: Optional[int] = None
    x_stop: Optional[int] = None
    y_start: Optional[int] = None
    y_stop: Optional[int] = None

    def __init__(self, configuration: VisualizerConfiguration) -> None:
        super().__init__()
        self.config = configuration
        self.network_handler = NetworkHandler(
            configuration.network_handler, configuration.network_loader
        )

        # configure window
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1800}x{950}")

        # configure grid layout (3x3)
        self.grid_columnconfigure((0, 1, 2), weight=1)
        # self.grid_rowconfigure((0, 1, 2), weight=1)

        # Add file browser button
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.grid(row=self._next_row(), column=0, columnspan=3)
        self.filedialog_textbox = ctk.CTkLabel(self.file_frame, text=NO_DATA_TEXT)
        self.filedialog_textbox.grid(row=0, column=0, padx=10)

        filedialog_button = ctk.CTkButton(
            self.file_frame,
            text="Select data folder...",
            command=self._change_directory,
        )
        filedialog_button.grid(row=0, column=1)

        self.bins_frame = ctk.CTkFrame(self)

        self.bins_frame.grid(row=self._next_row(), column=1)
        self.apply_bins_button = ctk.CTkButton(
            self.bins_frame, text="Apply bins", command=self._new_bins_value
        )
        self.apply_bins_button.grid(row=0, column=1, rowspan=4)

        # Start bin slider
        self.start_bin_slider_label = ctk.CTkLabel(
            self.bins_frame, text=START_BIN_INDEX_TEMPLATE.format(val=0)
        )
        self.start_bin_slider_label.grid(row=1, column=0)
        command = lambda val: self._slider_changed(
            self.start_bin_slider_label, val, START_BIN_INDEX_TEMPLATE
        )

        self.start_bin_var = ctk.IntVar(value=0)
        self.start_bin_slider = ctk.CTkSlider(
            self.bins_frame,
            from_=0,
            to=self.config.total_bins_in_video-1,
            variable=self.start_bin_var,
            width=300,
            command=command,
        )
        self.start_bin_slider.grid(row=2, column=0)
        # self.start_bin_slider.bind("<ButtonRelease-1>", self._new_bin_slider_value)

        # End bin slider
        self.end_bin_slider_label = ctk.CTkLabel(
            self.bins_frame, text=END_BIN_INDEX_TEMPLATE.format(val=0)
        )
        self.end_bin_slider_label.grid(row=3, column=0)
        command = lambda val: self._slider_changed(
            self.end_bin_slider_label, val, END_BIN_INDEX_TEMPLATE
        )

        self.end_bin_var = ctk.IntVar(value=0)
        self.end_bin_slider = ctk.CTkSlider(
            self.bins_frame,
            from_=0,
            to=self.config.total_bins_in_video-1,
            variable=self.end_bin_var,
            width=300,
            command=command,
        )
        self.end_bin_slider.grid(row=4, column=0)
        # self.end_bin_slider.bind("<ButtonRelease-1>", self._new_bin_slider_value)

        # Timestamp slider
        self.timestamp_slider_label = ctk.CTkLabel(
            self, text=TIMESTAMP_TEMPLATE.format(val=0.5)
        )
        self.timestamp_slider_label.grid(row=self._next_row(), column=1)
        self.timestamp_slider = ctk.CTkSlider(
            self, from_=0, to=1, width=300, command=self._timestamp_slider_changed
        )
        self.timestamp_slider.grid(row=self._next_row(), column=0, columnspan=3)
        self.timestamp_slider.bind("<ButtonRelease-1>", self._new_slider_value)

        # Zoom
        self.zoom_frame = ctk.CTkFrame(self)
        self.zoom_frame.grid(row = self._next_row(), column=0, columnspan=3)

        self.x_start_label = ctk.CTkLabel(self.zoom_frame, text="x_start")
        self.x_start_label.grid(row=0, column=0)
        self.x_start_string = ctk.StringVar(self)
        self.x_start_entry = ctk.CTkEntry(self.zoom_frame, textvariable=self.x_start_string)
        self.x_start_entry.grid(row=0, column=1, padx=2)

        self.x_stop_label = ctk.CTkLabel(self.zoom_frame, text="x_stop")
        self.x_stop_label.grid(row=0, column=2)
        self.x_stop_string = ctk.StringVar(self)
        self.x_stop_entry = ctk.CTkEntry(self.zoom_frame, textvariable=self.x_stop_string)
        self.x_stop_entry.grid(row=0, column=3, padx=2)

        self.y_start_label = ctk.CTkLabel(self.zoom_frame, text="y_start")
        self.y_start_label.grid(row=0, column=4)
        self.y_start_string = ctk.StringVar(self)
        self.y_start_entry = ctk.CTkEntry(self.zoom_frame, textvariable=self.y_start_string)
        self.y_start_entry.grid(row=0, column=5, padx=2)

        self.y_stop_label = ctk.CTkLabel(self.zoom_frame, text="y_stop")
        self.y_stop_label.grid(row=0, column=6)
        self.y_stop_string = ctk.StringVar(self)
        self.y_stop_entry = ctk.CTkEntry(self.zoom_frame, textvariable=self.y_stop_string)
        self.y_stop_entry.grid(row=0, column=7, padx=2)

        self.apply_zoom_button = ctk.CTkButton(self.zoom_frame, text="Apply zoom", command=self._apply_zoom)
        self.apply_zoom_button.grid(row=0,column=9)

        # Images
        self.start_image_label = ctk.CTkLabel(
            self, text="Image at start of bin", compound="bottom"
        )
        self.start_image_label.grid(row=self.row, column=0)

        self.end_image_label = ctk.CTkLabel(
            self, text="Image at end of bin", compound="bottom"
        )
        self.end_image_label.grid(row=self.row, column=2)

        # Prediction
        self.prediction_label = ctk.CTkLabel(
            self,
            text=f"Prediction at t={self.timestamp_slider._value:2f}",
            compound="bottom",
        )
        self.prediction_label.grid(row=self._next_row(), column=1)

        # GT
        self.gt_image_label = ctk.CTkLabel(
            self,
            text=GT_TEMPLATE.format(val=self.timestamp_slider._value),
            compound="bottom",
        )
        self.gt_image_label.grid(row=self.row, column=1)

        # Event image
        self.event_image_label = ctk.CTkLabel(
            self,
            text="Event polarity sum",
            compound="bottom",
        )
        self.event_image_label.grid(row=self.row, column=0)

    def run(self) -> None:
        self.mainloop()

    def _next_row(self) -> int:
        self.row += 1
        return self.row - 1

    def _change_directory(self) -> None:
        result = filedialog.askdirectory(initialdir="data/interim/coco/2subbins")
        result = NO_DATA_TEXT if isinstance(result, tuple) or result == "" else result
        self.data_directory = result
        self.filedialog_textbox.configure(text=result)
        self.filedialog_textbox.update()
        self.network_handler.set_data_directory(Path(self.data_directory))

    def _new_bins_value(self) -> None:
        self.network_handler.set_bin_indices(
            self.start_bin_var.get(), self.end_bin_var.get()
        )
        # Start and end
        self._update_start_and_end_images()
        # GT
        self._update_gt()
        # Events
        self._update_event_image()
        # Prediction
        self._update_prediction()

    def _update_start_and_end_images(self) -> None:
        start, end = self.network_handler.get_start_and_end_images()
        start = start[self.x_start:self.x_stop, self.y_start:self.y_stop]
        end = end[self.x_start:self.x_stop, self.y_start:self.y_stop]

        start_image = Image.fromarray(start * 255)
        end_image = Image.fromarray(end * 255)
        start_image = start_image.convert("RGB")
        end_image = end_image.convert("RGB")

        self._add_grid(start_image)
        self._add_grid(end_image)
        start_image = start_image.resize(IMAGE_SIZE, resample=Image.BOX)
        end_image = end_image.resize(IMAGE_SIZE, resample=Image.BOX)

        start_image = ctk.CTkImage(start_image, size=IMAGE_SIZE)
        end_image = ctk.CTkImage(end_image, size=IMAGE_SIZE)
        self.start_image_label.configure(image=start_image)
        self.start_image_label.image = start_image
        self.end_image_label.configure(image=end_image)
        self.end_image_label.image = end_image
    
    def _update_event_image(self) -> None:
        event_image = self.network_handler.get_event_image()
        event_image = event_image[self.x_start:self.x_stop, self.y_start:self.y_stop]
        self.event_image = Image.fromarray((event_image*255).astype(np.uint8))
        event_image = self.event_image.copy()

        self._add_grid(event_image)
        event_image = event_image.resize(IMAGE_SIZE, resample=Image.BOX)

        event_image = ctk.CTkImage(event_image, size=IMAGE_SIZE)
        self.event_image_label.configure(image=event_image)
        self.event_image_label.image = event_image


    def _update_gt(self) -> None:
        print(f"Updating GT to {self.timestamp_slider._value}")
        gt_image_np = self.network_handler.get_ground_truth(
            self.timestamp_slider._value, self.config.total_bins_in_video
        )

        gt_image_np = gt_image_np[self.x_start:self.x_stop, self.y_start:self.y_stop]
        gt_image = Image.fromarray(gt_image_np * 255).convert("RGB")
        self._add_grid(gt_image)
        gt_image = gt_image.resize(IMAGE_SIZE, resample=Image.BOX)

        gt_image = ImageTk.PhotoImage(gt_image, size=IMAGE_SIZE)
        self.gt_image_label.configure(
            image=gt_image,
            text=GT_TEMPLATE.format(val=self.timestamp_slider._value),
        )
        self.gt_image_label.image = gt_image

    def _add_grid(self, image: ImageTk) -> None:
        draw = ImageDraw.Draw(image)

        # Image dimensions
        width, height = image.size

        # Grid dimensions (number of rows and columns)

        # Drawing the grid
        for i, x in enumerate(range(0, width, GRID_LINES_PIXEL_SPACING)):
            fill = "red" if i % 2 == 0 else "blue"

            draw.line([(x, 0), (x, height)], fill=fill)

        for i, y in enumerate(range(0, height, GRID_LINES_PIXEL_SPACING)):
            fill = "red" if i % 2 == 0 else "blue"

            draw.line([(0, y), (width, y)], fill=fill)
            draw.line([(x, 0), (x, height)], fill=fill)


    def _timestamp_slider_changed(self, value: float) -> None:
        self.timestamp_slider_label.configure(text=TIMESTAMP_TEMPLATE.format(val=value))
        self.timestamp_slider_label.update()
        self._update_gt()

    def _new_slider_value(self, _: Any) -> None:
        self._update_prediction()
    
    def _update_prediction(self) -> None:
        print(f"Updating prediction to {self.timestamp_slider._value}")
        prediction = self.network_handler.get_reconstruction(self.timestamp_slider._value)
        print(f"Prediction range: {prediction.min()} to {prediction.max()}")
        prediction[prediction < 0] = 0
        prediction[prediction > 1] = 1

        prediction = prediction[self.x_start:self.x_stop, self.y_start:self.y_stop]
        prediction_image = Image.fromarray(prediction.numpy() * 255).convert("RGB")
        self._add_grid(prediction_image)
        prediction_image = prediction_image.resize(IMAGE_SIZE, resample=Image.BOX)

        prediction_image = ctk.CTkImage(prediction_image, size=IMAGE_SIZE)
        self.prediction_label.configure(
            image=prediction_image,
            text=PREDICTION_TEMPLATE.format(val=self.timestamp_slider._value),
        )
        self.prediction_label.image = prediction_image


    def _slider_changed(
        self, label: ctk.CTkLabel, value: float, template: str
    ) -> None:
        if self.end_bin_slider._value < self.start_bin_slider._value:
            if label is self.start_bin_slider_label:
                val = self.start_bin_var.get()
                self.end_bin_var.set(val)
                self.end_bin_slider.update()
                self.end_bin_slider._command(val)
            elif label is self.end_bin_slider_label:
                val = self.end_bin_var.get()
                self.start_bin_var.set(val)
                self.start_bin_slider.update()
                self.start_bin_slider._command(val)

        label.configure(text=template.format(val=value))
        label.update()
    

    def _new_bin_slider_value(self, _: Any) -> None:
        print(
            f"Start: {self.start_bin_slider._value*(self.config.total_bins_in_video-1):.0f}, end:"
            f" {self.end_bin_slider._value*(self.config.total_bins_in_video-1):.0f}"
        )

    def _apply_zoom(self) -> None:
        self.apply_zoom_button.focus_set()
        try:
            self.x_start = int(self.x_start_string.get())
            self.x_stop = int(self.x_stop_string.get())
            self.y_start = int(self.y_start_string.get())
            self.y_stop = int(self.y_stop_string.get())
            print("Integers parsed!")
            self._update_gt()
            self._update_prediction()
            self._update_event_image()
            self._update_start_and_end_images()
        except Exception:
            self.x_start = None
            self.x_stop = None
            self.y_start = None
            self.y_stop = None
            print("Integer parsing failed!")

if __name__ == "__main__":
    with open(
        "configs/interactive_visualizer/visualizer.yml", encoding="utf8"
    ) as infile:
        yaml = YAML().load(infile)
        config = VisualizerConfiguration.parse_obj(yaml)

    app = Visualizer(config)
    app.mainloop()
