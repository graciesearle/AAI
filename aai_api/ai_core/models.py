from django.db import models


class InferenceLog(models.Model):
    """
    ML telemetry and interaction log.
    Maintained inside AAI to satisfy academic case study requirements
    for end-user interaction tracking and user override handling loops.
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    producer_id = models.IntegerField()
    product_id = models.IntegerField(null=True, blank=True)
    model_version = models.CharField(max_length=64)
    confidence = models.FloatField()
    color_score = models.FloatField()
    size_score = models.FloatField()
    ripeness_score = models.FloatField()
    
    # What the model predicted initially
    predicted_grade = models.CharField(max_length=4)
    
    # The crucial retraining feedback loop variables (override endpoint)
    producer_accepted = models.BooleanField(null=True, blank=True)
    override_grade = models.CharField(max_length=4, blank=True)

    def __str__(self):
        return f"InferenceLog #{self.pk} - Producer {self.producer_id} - Grade {self.predicted_grade}"
