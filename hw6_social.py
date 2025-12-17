"""
15-110 F25 Hw6 - Social Media Analytics Project
Name:
AndrewID:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### WEEK 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]


'''
parseLabel(label)
#3 [Check6-1]
Parameters: str
Returns: dict mapping str to str
'''
def parseLabel(label):
    prefix = "From: " 
    if label.startswith(prefix):
        label = label[len(prefix):] #if the prefix begins with From: slice it off

    parenStart = label.find("(") #finds and retursn teh index of the first character (
    name = label[:parenStart].strip() #takes everything from the start up to but not inclduing (
    inside = label[parenStart + 1:].strip()#removes any extra spaces at the end

    if inside.startswith("("): #if the inside begins with "(" remove the first character
        inside = inside[1:]
    if inside.endswith(")"):#if the inside begins with ")" remove the last character
        inside = inside[:-1]
    
    splitToken = " from "
    fromIndex = inside.find(splitToken)
    position = inside[:fromIndex].strip() #take everything before the from part which is their position
    state = inside[fromIndex + len(splitToken):].strip() #take everything after the from part which is their state


    return {"name":name, "position": position, "state": state}




'''
getRegionFromState(stateDf, state)
#4 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    match = stateDf[stateDf["state"] == state] #keep only the rows where the "state" column matches the given state
    region = match.iloc[0]["region"] #grab the region from the first row of that filtered result
    return region



'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    hashTagList = []
    splitMessage = message.split(" ")
    for word in splitMessage:
        for i in range(len(word)):
            if word[i] == "#":
                for j in range(i + 1, len(word)):
                    if word[j] in endChars:
                        newWord = word[i:j]
                        hashTagList.append(newWord)
                        break
                else:
                    hashTagList.append(word[i:])
                    break
                         
    #print(hashTagList)
    return hashTagList

'''
findSentiment(classifier, message)
#6 [Check6-1]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)
    if score["compound"] > 0.1:
        return "positive"
    elif score["compound"] < 0.1 and score["compound"] > -0.1 :
        return "neutral"
    else:
        return "negative"

'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []
    sentiments = []
    sentClass = SentimentIntensityAnalyzer()
    for label in data["label"]:
        parsed = parseLabel(label)
        name = parsed["name"]
        position = parsed["position"]
        state = parsed["state"]
        region = getRegionFromState(stateDf, state)
        
        names.append(name)
        positions.append(position)
        states.append(state)
        regions.append(region)

    for text in data["text"]:
        hashtags.append(findHashtags(text))
        sentiments.append(findSentiment(sentClass, text))

    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions
    data["hashtags"] = hashtags
    data["sentiment"] = sentiments

    return


### WEEK 2 ###


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    stateCounts = {}

    if colName == "" and dataToCount == "":
        for i in range(len(data)):
            state = data.iloc[i]["state"]
            if state not in stateCounts:
                stateCounts[state] = 1
            else:
                stateCounts[state] += 1
        return stateCounts


    for i in range(len(data)):
        if data.iloc[i][colName] == dataToCount:
            state = data.iloc[i]["state"]

            if state not in stateCounts:
                stateCounts[state] = 1

            else:
                stateCounts[state] += 1

    return stateCounts


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    regionDict = {}

    for i in range(len(data)):
        region = data.iloc[i]["region"]
        value = data.iloc[i][colName]

        # create inner dictionary if region not seen
        if region not in regionDict:
            regionDict[region] = {}

        # count value inside that region
        if value not in regionDict[region]:
            regionDict[region][value] = 1
        else:
            regionDict[region][value] += 1

    return regionDict




'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtagCounts = {}
    for i in range(len(data)):
        hashtags = data.iloc[i]["hashtags"]
        for tag in hashtags:
            if tag not in hashtagCounts:
                hashtagCounts[tag] = 1
            else:
                hashtagCounts[tag] += 1

    return hashtagCounts

'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    topHashtags = {}

    while len(topHashtags) < count:
        maxTag = None
        maxCount = -1
        for tag in hashtags:
            if tag not in topHashtags:
                if hashtags[tag] > maxCount:
                    maxTag = tag
                    maxCount = hashtags[tag]
        topHashtags[maxTag] = maxCount

    return topHashtags



'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    totalScore = 0
    count = 0
    for i in range(len(data)):
        hashtags = data.iloc[i]["hashtags"]
        if hashtag in hashtags:
            sentiment = data.iloc[i]["sentiment"]
            if sentiment == "positive":
                totalScore += 1
            elif sentiment == "negative":
                totalScore -= 1
            else:
                totalScore += 0
            count += 1

    return totalScore / count


### WEEK 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    states = list(stateCounts.keys())
    counts = list(stateCounts.values())

    plt.bar(states, counts)
    plt.title(title)
    plt.xticks(rotation="vertical")
    plt.show()
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    # 1) Build rate dictionary
    stateRates = {}

    for state in stateFeatureCounts:
        total = stateCounts[state]
        feature = stateFeatureCounts[state]
        rate = feature / total
        stateRates[state] = rate

    # 2) Find top n by rate (like mostCommonHashtags)
    topStates = {}

    while len(topStates) < n:
        bestState = None
        bestRate = -1

        for state in stateRates:
            if state not in topStates:
                if stateRates[state] > bestRate:
                    bestRate = stateRates[state]
                    bestState = state

        topStates[bestState] = bestRate

    # 3) Graph
    graphStateCounts(topStates, title)

    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    # 1) Build feature list (xLabels)
    features = []
    for region in regionDicts:
        for feature in regionDicts[region]:
            if feature not in features:
                features.append(feature)

    # 2) Build region list (labelList)
    regions = []
    for region in regionDicts:
        regions.append(region)

    # 3) Build values list (valueLists)
    valueLists = []
    for region in regions:
        temp = []
        for feature in features:
            if feature in regionDicts[region]:
                temp.append(regionDicts[region][feature])
            else:
                temp.append(0)
        valueLists.append(temp)

    # 4) Graph it
    sideBySideBarPlots(features, regions, valueLists, title)
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    hashtagCounts = getHashtagRates(data)
    top50 = mostCommonHashtags(hashtagCounts, 50)

    hashtags = []
    frequencies = []
    sentiments = []

    for tag in top50:
        hashtags.append(tag)
        frequencies.append(top50[tag])
        sentiments.append(getHashtagSentiment(data, tag))

    title = "Hashtag Frequency vs Sentiment (Top 50 Hashtags)"
    scatterPlot(frequencies, sentiments, hashtags, title)

    return


#### WEEK 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()