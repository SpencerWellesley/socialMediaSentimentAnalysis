from hw6_social import *
import pandas as pd

#### WEEK 1 TESTS ####

def testParseLabel():
    print("Testing parseLabel()...", end="")
    assert(parseLabel("From: Steny Hoyer (Representative from Maryland)") == { "name" : "Steny Hoyer", "position" : "Representative", "state" : "Maryland" })
    assert(parseLabel("From: Mitch (Senator from Kentucky)") == { "name" : "Mitch", "position" : "Senator", "state" : "Kentucky" })
    assert(parseLabel("From: Heidi Heitkamp (Senator from North Dakota)") == { "name" : "Heidi Heitkamp", "position" : "Senator", "state" : "North Dakota" })
    assert(parseLabel("From: Chris Collins (Representative from New York)") == { "name" : "Chris Collins", "position" : "Representative", "state" : "New York" })
    assert(parseLabel("From: Martha Roby (Representative from Alabama)") == { "name" : "Martha Roby", "position" : "Representative", "state" : "Alabama" })
    assert(parseLabel("From: Kelly (Professor from PA)") == { "name" : "Kelly", "position" : "Professor", "state" : "PA" })
    assert(parseLabel("From: Stephanie Rosenthal (Prof from PA)") == { "name" : "Stephanie Rosenthal", "position" : "Prof", "state" : "PA" })
    print("... done!")

def testGetRegionFromState():
    print("Testing getRegionFromState()...", end="")
    stateDf = pd.read_csv("data/statemappings.csv")
    assert(str(getRegionFromState(stateDf, "California")) == "West")
    assert(str(getRegionFromState(stateDf, "Maine")) == "Northeast")
    assert(str(getRegionFromState(stateDf, "Nebraska")) == "Midwest")
    assert(str(getRegionFromState(stateDf, "Texas")) == "South")
    print("... done!")

def testFindHashtags():
    print("Testing findHashtags()...", end="")
    assert(findHashtags("I am so #excited to watch #TheMandalorian! #starwars") == [ "#excited", "#TheMandalorian", "#starwars" ])
    assert(findHashtags("#CMUCarnival will be amazing as long as it doesn't rain #weatherchannel") == [ "#CMUCarnival", "#weatherchannel" ])
    assert(findHashtags("#Whatif, #everything #is: #hashtags?") ==  [ "#Whatif", "#everything", "#is", "#hashtags" ])
    assert(findHashtags("I don't like hashtags, I think they're overused") == [ ])
    assert(findHashtags("So excited for #registration!Let's go CMU!") == [ "#registration" ])
    assert(findHashtags("I'm nervous-#registration but I think it should work out") == [ "#registration" ])
    assert(findHashtags("I'm waitlisted for everything #registration...") == [ "#registration" ])
    assert(findHashtags("Not sure what to take #110#112") == [ "#110", "#112" ])
    assert(findHashtags("Uh oh#") == ["#"])
    print("... done!")

def testFindSentiment():
    print("Testing findSentiment()...", end="")
    classifier = SentimentIntensityAnalyzer()
    assert(findSentiment(classifier, "great") == "positive")
    assert(findSentiment(classifier, "bad") == "negative")
    assert(findSentiment(classifier, "hello") == "neutral")
    assert(findSentiment(classifier, "Being a senator means getting votes on what you want and also having to take tough votes on what you don't") == "neutral")
    assert(findSentiment(classifier, "Had opportunity to participate in panel on importance of disaster mitigation") == "neutral")
    assert(findSentiment(classifier, "If you're in the area tomorrow make sure to stop by my office! I will be hold office hours from 3-4pm CST.") == "positive")
    assert(findSentiment(classifier, "The House will pass a bill to pay federal workers for their time in furlough once the shutdown ends.") == "negative")
    print("...done!")

def testAddColumns():
    print("Testing addColumns()...", end="")
    df = pd.read_csv("data/politicaldata.csv")
    stateDf = pd.read_csv("data/statemappings.csv")
    addColumns(df, stateDf)
    assert(df["name"][1] == "Mitch McConnell")
    assert(df["name"][4] == "Mark Udall")
    assert(df["name"][4979] == "Ted Yoho")
    assert(df["position"][1] == "Senator")
    assert(df["position"][4] == "Senator")
    assert(df["position"][4979] == "Representative")
    assert(df["state"][1] == "Kentucky")
    assert(df["state"][4] == "Colorado")
    assert(df["state"][4979] == "Florida")
    assert(df["region"][1] == "South")
    assert(df["region"][4] == "West")
    assert(df["region"][4979] == "South")
    assert(df["hashtags"][1] == [ "#Obamacare" ])
    assert(df["hashtags"][4] == [ "#drones", "#innovation", "#privacy", "#UAS" ])
    assert(df["hashtags"][4979] == [ ])
    assert(df["sentiment"][0] == "neutral")
    assert(df["sentiment"][1] == "negative")
    assert(df["sentiment"][4978] == "positive")
    print("... done!")

def week1Tests():
    testParseLabel()
    testGetRegionFromState()
    testFindHashtags()
    testFindSentiment()
    testAddColumns()

def runWeek1():
    df = pd.read_csv("data/politicaldata.csv")
    stateDf = pd.read_csv("data/statemappings.csv")
    addColumns(df, stateDf)
    print("Updated dataframe:")
    print(df)


#### WEEK 2 TESTS ####

def testGetDataCountByState(df):
    print("Testing getDataCountByState()...", end="")
    d1 = getDataCountByState(df, "sentiment", "negative")
    assert(len(d1) == 49)
    assert(d1["Pennsylvania"] == 48)
    assert(d1["North Dakota"] == 3)
    assert(d1["Louisiana"] == 20)

    d2 = getDataCountByState(df, "message", "attack")
    assert(len(d2) == 37)
    assert(d2["Pennsylvania"] == 9)
    assert(d2["Maryland"] == 4)
    assert(d2["Nevada"] == 1)

    d3 = getDataCountByState(df, "bias", "partisan")
    assert(len(d3) == 50)
    assert(d3["Pennsylvania"] == 40)
    assert(d3["Maryland"] == 44)
    assert(d3["Nevada"] == 10)

    d4 = getDataCountByState(df, "", "")
    assert(len(d4) == 50)
    assert(d4["Pennsylvania"] == 177)
    assert(d4["Maryland"] == 108)
    assert(d4["Nevada"] == 50)
    print("... done!")

def testGetDataForRegion(df):
    print("Testing getDataForRegion()...", end="")
    d1 = getDataForRegion(df, "message")
    assert(len(d1) == 4)
    assert(len(d1["South"]) == 9)
    assert(d1["South"]["policy"] == 563)
    assert(d1["Northeast"]["attack"] == 23)

    d2 = getDataForRegion(df, "audience")
    assert(len(d2) == 4)
    assert(len(d2["South"]) == 2)
    assert(d2["South"]["national"] == 1561)
    assert(d2["Midwest"]["constituency"] == 265)
    assert(d2["Northeast"]["national"] == 682)

    print(".. done!")

def testGetHashtagRates(df):
    print("Testing getHashtagRates()...", end="")
    d = getHashtagRates(df)
    assert(len(d) == 1526)
    assert(d["#TrainWreck"] == 8)
    assert(d["#jobs"] == 20)
    assert(d["#STEM"] == 5)
    assert(d["#ObamaCare"] == 20)
    print("... done!")

def testMostCommonHashtags(df):
    print("Testing mostCommonHashtags()...", end="")
    d1 = { "#CMU" : 10, "#TheMandalorian" : 15, "#tgif" : 3, "#homework" : 20, "#hashtag" : 1, "#programming" : 7, "#testcase" : 1, "#WorldPeace" : 9, "#coffee" : 18, "#naptime" : 2 }
    assert(mostCommonHashtags(d1, 1) == { "#homework" : 20 })
    assert(mostCommonHashtags(d1, 2) == { "#homework" : 20, "#coffee" : 18 })
    assert(mostCommonHashtags(d1, 5) == { "#homework" : 20, "#coffee" : 18, "#TheMandalorian" : 15, "#CMU" : 10, "#WorldPeace" : 9 })

    d2 = getHashtagRates(df)
    assert(mostCommonHashtags(d2, 1) == { "#Obamacare" : 61 })
    assert(mostCommonHashtags(d2, 7) == { "#Obamacare" : 61, "#IRS" : 26, "#RenewUI" : 21, "#jobs" : 20, "#Benghazi" : 20, "#ObamaCare" : 20, "#SOTU" : 20 })
    print("... done!")

def testGetHashtagSentiment(df):
    # Note - we're comparing floats here, so we'll check if they're
    # almost equal instead of exactly equal
    print("Testing getHashtagSentiment()...", end="")
    import math
    assert(math.isclose(getHashtagSentiment(df, "#TrainWreck"), -0.125))
    assert(math.isclose(getHashtagSentiment(df, "#jobs"), 0.7894736842105263))
    assert(math.isclose(getHashtagSentiment(df, "#STEM"), 0.6))
    assert(math.isclose(getHashtagSentiment(df, "#ObamaCare"), 0))
    print("... done!")


def week2Tests():
    df = pd.read_csv("data/politicaldata.csv")
    stateDf = pd.read_csv("data/statemappings.csv")
    addColumns(df, stateDf)

    testGetDataCountByState(df)
    testGetDataForRegion(df)

    testGetHashtagRates(df)
    testMostCommonHashtags(df)
    testGetHashtagSentiment(df)

def runWeek2():
    df = pd.read_csv("data/politicaldata.csv")
    stateDf = pd.read_csv("data/statemappings.csv")
    addColumns(df, stateDf)

    stateCounts = getDataCountByState(df, "", "")
    print("Total Messages Per State")
    print(stateCounts)

    negSentiments = getDataCountByState(df, "sentiment", "negative")
    print("State Counts for Negative Sentiment")
    print(negSentiments)
    attacks = getDataCountByState(df, "message", "attack")
    print("\nState Counts for Attacks")
    print(attacks)
    partisanship = getDataCountByState(df, "bias", "partisan")
    print("\nState Counts for Partisanship")
    print(partisanship)

    messages = getDataForRegion(df, "message")
    print("\nMessage Types for Region")
    print(messages)
    audiences = getDataForRegion(df, "audience")
    print("\nAudience Types for Region")
    print(audiences)

    hashtags = getHashtagRates(df)
    freqHashtags = mostCommonHashtags(hashtags, 6)
    for hashtag in freqHashtags:
        print(hashtag, "sentiment score:", getHashtagSentiment(df, hashtag))


### WEEK 3 TESTS ###

# Instead of running individual tests, check the new graph generated by doWeek3
# after you finish each function.
def runWeek3():
    print("Prepare for a bunch of charts!")
    df = pd.read_csv("data/politicaldata.csv")
    stateDf = pd.read_csv("data/statemappings.csv")
    addColumns(df, stateDf)

    stateCounts = getDataCountByState(df, "", "")

    print("Basic bar charts:")
    twitterCounts = getDataCountByState(df, "source", "facebook")
    graphStateCounts(stateCounts, "Total Messages Per State")
    graphStateCounts(twitterCounts, "Total *Facebook* Messages Per State")

    print("Filtered bar charts:")
    attackCounts = getDataCountByState(df, "message", "attack")
    policyCounts = getDataCountByState(df, "message", "policy")
    nationalCounts = getDataCountByState(df, "audience", "national")
    graphTopNStates(stateCounts, attackCounts, 5, "Top Attack Message Rates")
    graphTopNStates(stateCounts, policyCounts, 5, "Top Policy Message Rates")
    graphTopNStates(stateCounts, nationalCounts, 5, "Top National Message Rates")

    print("Side-by-side bar charts:")
    messageTypes = getDataForRegion(df, "message")
    positionTypes = getDataForRegion(df, "position")
    graphRegionComparison(messageTypes, "Messages by Region")
    graphRegionComparison(positionTypes, "Position by Region")

    print("Scatterplot:")
    graphHashtagSentimentByFrequency(df)