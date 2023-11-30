from django.http import JsonResponse
from drf_spectacular.utils import extend_schema
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView

from LotteryGuesserV2.pagination import CustomPagination
from algorithms.models import lg_lottery_type, lg_lottery_winner_number

from lottery_handler.serializers import GetLotteryNumbersWithAlgorithm, LotteryNumbers, LotteryAlgorithmSerializer
from lottery_handler.signals import list_processor_files, call_get_numbers_dynamically


class LotteryNumbersApiView(APIView,CustomPagination):
    permission_classes = (AllowAny,)
    pagination_class = CustomPagination

    @extend_schema(
        summary="Get Lottery Numbers",
        parameters=[GetLotteryNumbersWithAlgorithm],
        responses={
            200: LotteryNumbers(many=True), })
    @action(methods=['GET'], detail=True, pagination_class=CustomPagination)
    def get(self, request):
        serializer = GetLotteryNumbersWithAlgorithm(data=request.GET)
        serializer.is_valid(raise_exception=True)

        lottery_type_id = serializer.validated_data["lottery_type_id"]
        lottery_type = lg_lottery_type.objects.filter(id=lottery_type_id).first()
        if lottery_type is None:
            return JsonResponse({"error": "Item not found"}, status=404)


        response = []
        result = call_get_numbers_dynamically(lottery_type)

        for key, value in result.items():

            data = {
                "numbers": value,
                'algorithms': key
            }
            response_serializer = LotteryNumbers(data=data)
            response_serializer.is_valid(raise_exception=True)
            response.append(response_serializer.data)
        results = self.paginate_queryset(response, request, view=self)
        return self.get_paginated_response(results)


class LotteryAlgorithmsApiView(APIView):
    permission_classes = (AllowAny,)

    @extend_schema(
        summary="Get Lottery Algorithms",
        responses={
            200: LotteryAlgorithmSerializer(many=True), })

    def get(self, request):
        response = []
        result = list_processor_files()

        for item in result:
            data = {
                "algorithm_type": item
            }

            response_serializer = LotteryAlgorithmSerializer(data=data)
            response_serializer.is_valid(raise_exception=True)
            response.append(response_serializer.data)
        return JsonResponse(response, safe=False)


