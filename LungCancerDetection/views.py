from __future__ import print_function
from LungCancerDetection.ml_models import prediction
from LungCancerDetection.settings import BASE_DIR
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from PIL import Image
from io import BytesIO
import numpy


class DetectCancer(APIView):
    permission_classes = []
    parser_classes = [FileUploadParser]

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['file']
        readed_file = file_obj.read()
        in_mem = BytesIO(readed_file)
        image = Image.open(in_mem)
        pix = numpy.array(image)
        model_path = BASE_DIR / 'LungCancerDetection/my_model.h5'
        result = prediction(model_path,pix)
        final_result = float("{:.2f}".format(result))
        return Response({"message": "success","percentage":final_result}, status=200)
