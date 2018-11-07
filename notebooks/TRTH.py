#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Damien Challet
"""

import glob 
import re
import pandas as pd

def loadTRTHEvents(myfile):
    PD =pd.read_csv(myfile)
    PD.columns=['date','second','bid', 'bidQ', 'ask', 'askQ', 'last', 'lastQ']
    PD["date"] = pd.to_datetime(PD["date"]*86400-2209161600+PD["second"], unit="s")
    if re.search("\.(PA|VX|S)_events",myfile):   #Paris and Swiss exchange
        myTZ="Europe/Paris"
        PD=PD[PD["second"]>=9*3600]
        PD=PD[PD["second"]<=17.5*3600]

    if re.search("\.(OQ|N|O|A|P)_events",myfile):   #OQ: Nasdaq, N: NYSE, P: NYSE Arca, A: AMEX
        myTZ="US/Eastern"
        PD=PD[PD["second"]>=9.5*3600]
        PD=PD[PD["second"]<=16*3600]
    
    
    PD["date"] = PD.date.dt.tz_localize(myTZ)
    PD.set_index(PD["date"],inplace=True)
    PD.drop("date",axis=1,inplace=True)
    PD.drop("second",axis=1,inplace=True)

    return(PD)
        
