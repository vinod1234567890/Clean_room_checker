import sys
import os
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import PlainTextResponse
from starlette.responses import JSONResponse
import uvicorn
from starlette.templating import Jinja2Templates
import os
from PIL import Image
import io
import json

# programmer defined libraries.

from similarity import *

# getting all the templets for the following dir.
templates = Jinja2Templates(directory="templates")


async def index(request):
    if request.method == "POST":
        form = await request.form()
        file = form["files[]"].file
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        if file:
            loc = os.path.join(
                os.getcwd(),
                "dataset",
                "Images",
            )
            # checking whether the dic exist.
            if not os.path.isdir(loc):
                os.mkdir(loc)
            # deleting the existing files in the folder.
            if len(os.listdir(loc)) > 0:
                for f in os.listdir(loc):
                    os.remove(os.path.join(loc, f))
            # saving the file.
            image1 = image.save(os.path.join(loc, form["files[]"].filename))

            # getting the similarity score
            score = get_cosine_similarity()
            context = {"data": str(score)}
            return JSONResponse(context)
            # return templates.TemplateResponse("index.html", context)
    else:
        return templates.TemplateResponse(
            "index.html", {"request": request, "data": ""}
        )


# All the routs of this website.
routes = [
    Route("/", index, methods=["GET", "POST"]),
]
# App congiguration.
app = Starlette(debug=True, routes=routes)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
