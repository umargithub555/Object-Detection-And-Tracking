from fastapi import APIRouter, Query
from app.models.detection_record import DetectionRecord
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/stats")
async def get_dashboard_stats():
    """
    Returns aggregated detection stats:
    - Total detections (lifetime)
    - Detections in the last 24 hours
    - Breakdown by major classes
    """
    now = datetime.utcnow()
    last_24h = now - timedelta(days=1)

    # Lifetime stats
    all_records = await DetectionRecord.find_all().to_list()
    total_counts = defaultdict(int)
    for record in all_records:
        for cls, count in record.counts.items():
            total_counts[cls] += count

    # Last 24h stats
    recent_records = await DetectionRecord.find(DetectionRecord.timestamp >= last_24h).to_list()
    recent_counts = defaultdict(int)
    for record in recent_records:
        for cls, count in record.counts.items():
            recent_counts[cls] += count

    return {
        "lifetime": {
            "total": sum(total_counts.values()),
            "breakdown": dict(total_counts)
        },
        "last_24h": {
            "total": sum(recent_counts.values()),
            "breakdown": dict(recent_counts)
        }
    }

@router.get("/trends")
async def get_detection_trends():
    """
    Returns hourly detection trends for the last 24 hours.
    """
    now = datetime.utcnow()
    last_24h = now - timedelta(days=0)

    # Aggregate records by hour
    records = await DetectionRecord.find(DetectionRecord.timestamp >= last_24h).sort(+DetectionRecord.timestamp).to_list()
    
    # Initialize trend map with 24 hours
    trends = {}
    for i in range(24):
        hour_dt = (last_24h + timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
        hour_key = hour_dt.strftime("%Y-%m-%d %H:00")
        trends[hour_key] = defaultdict(int)

    for record in records:
        hour_key = record.timestamp.strftime("%Y-%m-%d %H:00")
        if hour_key in trends:
            for cls, count in record.counts.items():
                trends[hour_key][cls] += count

    # Convert to a sorted list for the frontend
    sorted_trends = []
    for hour in sorted(trends.keys()):
        sorted_trends.append({
            "hour": hour,
            "counts": dict(trends[hour])
        })

    return sorted_trends

@router.get("/activity")
async def get_recent_activity(limit: int = Query(10, ge=1, le=100)):
    """
    Returns the most recent detection activities.
    """
    records = await DetectionRecord.find_all().sort(-DetectionRecord.timestamp).limit(limit).to_list()
    
    activity = []
    for r in records:
        activity.append({
            "id": str(r.id),
            "timestamp": r.timestamp,
            "counts": r.counts,
            "filename": r.filename,
            "duration": r.duration,
            "processing_time": r.processing_time
        })
        
    return activity
