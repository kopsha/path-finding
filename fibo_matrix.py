from PIL import Image, ImageDraw


# create an image
out = Image.new("RGB", (1024, 1024), (255, 255, 255))

draw = ImageDraw.Draw(out)
draw.line((0, 0) + out.size, fill=128)
draw.line((0, out.size[1], out.size[0], 0), fill=128)

# write to stdout
out.save("test.png")
