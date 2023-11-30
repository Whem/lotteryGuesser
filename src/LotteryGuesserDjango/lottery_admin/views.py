from django.http import JsonResponse
from drf_spectacular.utils import extend_schema
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView

from LotteryGuesserV2.pagination import CustomPagination
from algorithms.models import lg_lottery_type
from common_tools.serializers import SuccessSerializer
from lottery_admin.serializers import PostLotteryTypeSerializers, FillLotteryTypeWithWinNumbers
from lottery_admin.signals import download_numbers_from_internet


# Create your views here.
class LotteryTypeApiView(APIView,CustomPagination):
    permission_classes = (AllowAny,)
    pagination_class = CustomPagination

    @extend_schema(
        summary="Get Lottery Type",

        responses={
            200: PostLotteryTypeSerializers(many=True), })
    @action(methods=['GET'], detail=True, pagination_class=CustomPagination)
    def get(self, request):
        lottery_types = lg_lottery_type.objects.all()

        lottery_array = []
        for lottery_type in lottery_types:
            serializer = PostLotteryTypeSerializers(data=lottery_type.__dict__)
            if serializer.is_valid():
                lottery_array.append(serializer.data)

        results = self.paginate_queryset(lottery_array, request, view=self)
        return self.get_paginated_response(results)

    @extend_schema(
        summary="Post Lottery Type",
        request=PostLotteryTypeSerializers,
        responses={
            200: PostLotteryTypeSerializers})
    def post(self, request, ):
        serializer = PostLotteryTypeSerializers(data=request.data)
        serializer.is_valid(raise_exception=True)

        if "id" in serializer.validated_data and serializer.validated_data["id"] is not None:
            item = lg_lottery_type.objects.filter(id=serializer.validated_data["id"]).first()
            if item is None:
                return JsonResponse({"error": "Item not found"}, status=404)

        else:
            item = lg_lottery_type()

        for key, value in serializer.validated_data.items():
            if hasattr(item, key):
                setattr(item, key, value)
        item.save()

        serializer = PostLotteryTypeSerializers(data=item.__dict__)
        if serializer.is_valid():
            return JsonResponse(serializer.data, safe=False)
        else:
            return JsonResponse(serializer.errors, status=400)


class FillLotteryTypeApiView(APIView, CustomPagination):
    permission_classes = (AllowAny,)
    pagination_class = CustomPagination

    @extend_schema(
        summary="Fill Lottery Type",
        request=FillLotteryTypeWithWinNumbers,
        responses={
            200: SuccessSerializer})
    def post(self, request, ):
        serializer = FillLotteryTypeWithWinNumbers(data=request.data)
        serializer.is_valid(raise_exception=True)
        item = None
        if "lottery_type_id" in serializer.validated_data and serializer.validated_data["lottery_type_id"] is not None:
            item = lg_lottery_type.objects.filter(id=serializer.validated_data["lottery_type_id"]).first()
            if item is None:
                return JsonResponse({"error": "Item not found"}, status=404)

        download_numbers_from_internet(item)

        return JsonResponse({"success": True}, safe=False)

