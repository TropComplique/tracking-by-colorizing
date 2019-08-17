I train the network using a random subset of [ActivityNet dataset](http://activity-net.org).

# How videos are downloaded
I downloaded the worst video quality using a command like this:
`youtube-dl -f worst -f mp4 "https://www.youtube.com/watch?v=VIDEO_ID" -o "videos/v_VIDEO_ID.mp4"`

# Some data statistics
1. Number of videos is 5896.
2. Total length is ~200 hours.
3. All videos have length at least 20 seconds.
4. Quantiles are 25%: 68, 50%: 123, 75%: 180.
