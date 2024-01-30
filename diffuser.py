#diffuser.py

from diffusers import DiffusionPipeline, StableDiffusionPipeline

model = "stabilityai/stable-diffusion-2-1"
#model = "runwayml/stable-diffusion-v1-5"
#model="openskyml/lexica-aperture-v3-5"

pipeline = DiffusionPipeline.from_pretrained(model, use_safetensors=True, safety_checker=None)

#pipeline = StableDiffusionPipeline.from_pretrained(model, use_safetensors=True, safety_checker=None)


pipeline.to("mps")

# def dummy(images, **kwargs):
#     return images, False

file_inc=1

pipeline.safety_checker = None
in_file_inc = 1

while True:
	print("Prompt: ")
	prompt = str(input())
	images = pipeline(prompt)
	print(images)
	for img in images[0]:
		img.save("test_image_" + str(file_inc) + "_" + str(in_file_inc) + ".png")
		in_file_inc += 1

	file_inc +=1 
	in_file_inc = 1




#image[0][0].save('nsfw.png')