"""
    @author   Thomas Lehmann
    @file     dateBack.py
    @brief    provides a human readable format for a time delta
"""
from datetime import datetime , timedelta

def dateBack(theDateAndTime, precise=False, fromDate=None):
    """ provides a human readable format for a time delta
        @param theDateAndTime this is time equal or older than now or the date in 'fromDate'
        @param precise        when true then milliseconds and microseconds are included
        @param fromDate       when None the 'now' is used otherwise a concrete date is expected
        @return the time delta as text

        @note I don't calculate months and years because those varies (28,29,30 or 31 days a month
              and 365 or 366 days the year depending on leap years). In addition please refer
              to the documentation for timedelta limitations.
    """
    if not fromDate:
        fromDate = datetime.now()

    if theDateAndTime > fromDate:    return None
    elif theDateAndTime == fromDate: return "now"

    delta = fromDate - theDateAndTime

    # the timedelta structure does not have all units; bigger units are converted
    # into given smaller ones (hours -> seconds, minutes -> seconds, weeks > days, ...)
    # but we need all units:
    deltaMinutes      = delta.seconds // 60
    deltaHours        = delta.seconds // 3600
    deltaMinutes     -= deltaHours * 60
    deltaWeeks        = delta.days    // 7
    deltaSeconds      = delta.seconds - deltaMinutes * 60 - deltaHours * 3600
    deltaDays         = delta.days    - deltaWeeks * 7
    deltaMilliSeconds = delta.microseconds // 1000
    deltaMicroSeconds = delta.microseconds - deltaMilliSeconds * 1000

    valuesAndNames =[ (deltaWeeks  ,"week"  ), (deltaDays   ,"day"   ),
                      (deltaHours  ,"hour"  ), (deltaMinutes,"minute"),
                      (deltaSeconds,"second") ]
    if precise:
        valuesAndNames.append((deltaMilliSeconds, "millisecond"))
        valuesAndNames.append((deltaMicroSeconds, "microsecond"))

    text =""
    for value, name in valuesAndNames:
        if value > 0:
            text += len(text)   and ", " or ""
            text += "%d %s" % (value, name)
            text += (value > 1) and "s" or ""

    # replacing last occurrence of a comma by an 'and'
    if text.find(",") > 0:
        text = " and ".join(text.rsplit(", ",1))

    return text

if __name__ == '__main__':
    fromDate = datetime(year=2012, month=4, day=26, hour=8, minute=40, second=45)
    print(dateBack(fromDate-timedelta(hours=1, seconds=1)))
