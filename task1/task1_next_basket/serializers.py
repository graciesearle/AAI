from rest_framework import serializers

class NextBasketRequestSerializer(serializers.Serializer):
    customer_id = serializers.IntegerField(help_text="The ID of the customer to predict for")
    top_n = serializers.IntegerField(default=5, required=False)

class NextBasketItemSerializer(serializers.Serializer):
    product_id = serializers.IntegerField()
    product_name = serializers.CharField()
    confidence = serializers.FloatField()
    reorder_probability = serializers.CharField()

class NextBasketResponseSerializer(serializers.Serializer):
    recommendations = NextBasketItemSerializer(many=True)
