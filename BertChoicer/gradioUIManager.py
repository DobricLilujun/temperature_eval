import gradio as gr

class GradioUIManager:
    """
    A class to manage and encapsulate the Gradio UI logic.
    """

    def __init__(self, choices, on_button_click, on_experiment_button_click):
        """
        Initializes the GradioUIManager.

        Args:
            choices (list): List of model choices for the Radio element.
            on_button_click (function): Callback function for the "Analyze" button.
            on_experiment_button_click (function): Callback function for the "Start Experiment" button.
        """
        self.choices = choices
        self.on_button_click = on_button_click
        self.on_experiment_button_click = on_experiment_button_click
        self.demo = None
        self.section_title = "Temperature Choicer"
        self.section_description = "This module facilitates the input of a prompt for analysis and processing. Based on the provided prompt and the selected model, the optimal temperature setting will be determined."
        self.recommendation_title = "Recommendation section"
        self.recommendation_description = "Provide the best temperature settings based on your input or prompts."
        self.section_title_size = 50
        self.section_description_size = 25
        self.subsection_title_size = 20
        self.subsection_description_size = 12
    def create_interface(self):
        """
        Creates the Gradio interface using the provided callback functions and model choices.
        """
        with gr.Blocks() as demo:
            # Section 1: Input and Analysis
            gr.HTML(
            "<div style='text-align: center; display: flex; align-items: center;'>"
            f"""
            <span style='color: gray; font-size: {self.section_title_size}px; font-weight: bold; display: inline;'>
                {self.section_title}
            </span>
            <span style='display: inline; vertical-align: middle; margin-left: 10px;'>
                <img src='https://raw.githubusercontent.com/DobricLilujun/imagesAll/main/imagesthermometer.png' 
                    alt='Thermometer Image' 
                    style='height: {self.section_title_size}px;'>
            </span>
            </div>
            """
            f"<div> <span style='color: gray; font-size: {self.section_description_size}px; display: inline-block;'><small>{self.section_description}</small></span></div>"
            ""
            )
            with gr.Row():
                with gr.Column(scale=0.5):
                    self.input_text_box = gr.Textbox(label="Input Your Prompt Here", elem_id="Input Prompt")
                    self.input_file_box = gr.File(
                        label="Input File",
                        elem_id="input_file",
                        elem_classes="small-input-file",
                    )

                    self.input_model_radio = gr.Radio(
                        choices=self.choices,
                        label="Select Your Model ",
                        info="Choose one, or select a similar size or smaller type (e.g., for Llama 2, choose from the Llama 2 series).",
                        elem_id="input_model",
                        value=self.choices[0],
                        interactive=True,
                    )
                with gr.Column(scale=0.5):
                    self.cls_label_output_box = gr.Plot(
                        label="Class Label Output (Bar Chart)",
                        elem_id="cls_label_output",
                    )

                    self.best_temperature_output_box = gr.Textbox(
                        label="Best Temperature As An Output",
                        elem_id="best_temperature_output",
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column(scale=1.0):
                    self.input_button = gr.Button("Analyze", elem_id="input_button")

            gr.HTML(
                "<div style='text-align: left;'>"
                f"<span style='color: gray; font-size: {self.subsection_title_size}px; font-weight: bold;'>{self.recommendation_title}</span><br>"
                f"<span style='color: gray; font-size: {self.subsection_description_size}px;'><small>{self.recommendation_description}</small></span>"
                "</div>"
            )
            gr.HTML("<hr>")

            with gr.Row():
                with gr.Column(scale=1):
                    self.input_temperature_slider = gr.Slider(
                        minimum=0,
                        maximum=2.0,
                        step=0.1,
                        label="Input Temperature",
                        elem_id="input_temperature",
                    )
                    self.input_api_box = gr.Textbox(
                        label="API Key",
                        elem_id="input_api",
                        value="http://localhost:11434/api/generate",
                    )
                    self.start_experiment_button = gr.Button(
                        "Start Experiment", elem_id="start_experiment"
                    )

            # Section 3: Outputs
            gr.HTML(
                "<div style='text-align: left;'>"
                f"<span style='color: gray; font-size: {self.subsection_title_size}px; font-weight: bold;'>Outputs</span><br>"
                f"<span style='color: gray; font-size: {self.subsection_description_size}px;'><small>The output files and analysis results.</small></span>"
                "</div>"
            )
            gr.HTML("<hr>")
            with gr.Row():
                self.output_text_box = gr.Textbox(
                    label="Output Text", elem_id="output_text", lines=10, interactive=False
                )
                self.download_results = gr.Button("Download Results", elem_id="download_results")

            # Button Click Events
            self.input_button.click(
                self.on_button_click,
                inputs=[self.input_text_box, self.input_file_box, self.input_model_radio],
                outputs=[
                    self.cls_label_output_box,
                    self.best_temperature_output_box,
                    self.input_temperature_slider,
                ],
            )

            self.start_experiment_button.click(
                self.on_experiment_button_click,
                inputs=[
                    self.input_text_box,
                    self.input_temperature_slider,
                    self.input_model_radio,
                    self.input_api_box,
                ],
                outputs=[self.output_text_box],
            )
        
        self.demo = demo

    def launch(self, share=False):
        """
        Launches the Gradio application.

        Args:
            share (bool): Whether to enable sharing the interface publicly.
        """
        if not self.demo:
            raise Exception("Interface not created. Call create_interface() first.")
        self.demo.launch(share=share)
