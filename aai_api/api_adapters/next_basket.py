from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
import logging

from task1.task1_next_basket.runtime import predict_next_basket
from task1.task1_next_basket.serializers import (
    NextBasketRequestSerializer,
    NextBasketResponseSerializer,
)

logger = logging.getLogger(__name__)

class NextBasketAdapterView(APIView):
    """
    Adapter for Task 1: Next Basket Prediction.
    Uses the LSTM sequence model to predict what a customer will buy in their next order.
    """
    def post(self, request):
        serializer = NextBasketRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        customer_id = serializer.validated_data['customer_id']
        top_n = serializer.validated_data.get('top_n', 5)

        try:
            results = predict_next_basket(user_id=customer_id, top_n=top_n)
            
            if isinstance(results, dict) and "error" in results:
                return Response({"detail": results["error"]}, status=status.HTTP_400_BAD_REQUEST)

            output = NextBasketResponseSerializer({"recommendations": results})
            return Response(output.data, status=status.HTTP_200_OK)
            
        except Exception as exc:
            logger.error(f"Next Basket Prediction failed: {exc}")
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
