from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import re

from mrt.e2e.meta import (
    NAME_RE, NEAR_RE, NAMES_TOKENIZED, NEARS_TOKENIZED, LEXICON,
)
from mrt.e2e.mr_utils import is_placeholder


def tag_tokens(tokens, **kwargs):
    tokens = [t if is_placeholder(t) else t.lower() for t in tokens]
    tags = ['0'] * len(tokens)

    for i, t in enumerate(tokens):

        if t == 'NAME':
            tags[i] = 'name=PLACEHOLDER'
        if t == 'NEAR':
            tags[i] = 'near=PLACEHOLDER'

        if t == 'zizzis':
            tags[i] = 'name=Zizzi'
        if t == 'chinese':
            tags[i] = 'food=Chinese'
        if t == 'chines':
            tags[i] = 'food=Chinese'
        if t == 'english':
            tags[i] = 'food=English'
        if t == 'british':
            tags[i] = 'food=English'
        if t == 'french':
            tags[i] = 'food=French'
        if t == 'indian':
            tags[i] = 'food=Indian'
        if t == 'italian':
            tags[i] = 'food=Italian'
        if t == 'japanese':
            tags[i] = 'food=Japanese'
        if t == 'oriental':
            tags[i] = 'food=Japanese'
        if tokens[i-1:i+1] == ['fast', 'food']:
            tags[i-1] = 'food=Fast food'
            tags[i] = 'food=Fast food'
        if tokens[i-2:i+1] == ['fast', '-', 'food']:
            tags[i-2] = 'food=Fast food'
            tags[i-1] = 'food=Fast food'
            tags[i] = 'food=Fast food'
        for near, nt in NEARS_TOKENIZED:
            if tokens[i-len(nt)+1:i+1] == nt:
                for j in range(i-len(nt)+1,i+1): 
                    tags[j] = f'near={near}'
 
        for name, nt in NAMES_TOKENIZED:
            if tokens[i-len(nt)+1:i+1] == nt:
                for j in range(i-len(nt)+1,i+1): 
                    tags[j] = f'name={name}'
 
        if t == 'pub':
            tags[i] = 'eat_type=pub'
        if t == 'restaurant':
            tags[i] = 'eat_type=restaurant'
        if t == 'coffee':
            tags[i] = 'eat_type=coffee shop'
        if tokens[i-1:i+1] == ['coffee', 'shop']:
            tags[i-1] = 'eat_type=coffee shop'
            tags[i] = 'eat_type=coffee shop'
        if tokens[i-2:i+1] == ['coffee', '-', 'shop']:
            tags[i-2] = 'eat_type=coffee shop'
            tags[i-1] = 'eat_type=coffee shop'
            tags[i] = 'eat_type=coffee shop'




 
        if tokens[i-4:i+1] == ['next', 'to', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-4:i+1] == ['near', 'to', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-4:i+1] == ['near', 'of', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-4:i+1] == ['close', 'to', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-9:i+1] == ['nearby', 'the', 'city', 'centre', 'and', 'the', 'restaurant', 'the', 'rice', 'boat']:
            tags[i] = 'near=The Rice Boat'
            tags[i-1] = 'near=The Rice Boat'
            tags[i-2] = 'near=The Rice Boat'

        if tokens[i-3:i+1] == ['in', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-3:i+1] == ['near', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-3:i+1] == ['beside', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'

        if tokens[i-3:i+1] == ['by', 'the', 'rice', 'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'


        for x1 in [['city'], ['town'], ['shopping']]:
            for x2 in [[], ['-'],]:
                for x3 in [['center'], ['centre']]:
                    p = x1 + x2 + x3
                    s = i - len(p) + 1
                    if tokens[s:i+1] == p:
                        for j in range(s,i+1):
                            tags[j] = 'area=city centre'

#
#        if tokens[i-1:i+1] == ['city', 'center']:
#            tags[i-1] = 'area=city centre'
#            tags[i] = 'area=city centre'
#        if tokens[i-1:i+1] == ['city', 'centre']:
#            tags[i-1] = 'area=city centre'
#            tags[i] = 'area=city centre'
#        if tokens[i-1:i+1] == ['town', 'center']:
#            tags[i-1] = 'area=city centre'
#            tags[i] = 'area=city centre'
#        if tokens[i-1:i+1] == ['town', 'centre']:
#            tags[i-1] = 'area=city centre'
#            tags[i] = 'area=city centre'
#        if tokens[i-1:i+1] == ['shopping', 'centre']:
#            tags[i-1] = 'area=city centre'
        if tokens[i-3:i+1] == ['centre', 'of', 'the', 'city']:
            tags[i-3] = 'area=city centre'
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-3:i+1] == ['centre', 'of', 'the', 'town']:
            tags[i-3] = 'area=city centre'
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-3:i+1] == ['center', 'of', 'the', 'city']:
            tags[i-3] = 'area=city centre'
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-3:i+1] == ['center', 'of', 'the', 'town']:
            tags[i-3] = 'area=city centre'
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-2:i+1] == ['centre', 'of', 'city']:
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-2:i+1] == ['centre', 'of', 'town']:
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-2:i+1] == ['center', 'of', 'city']:
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-2:i+1] == ['center', 'of', 'town']:
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-2:i+1] == ['in', 'the', 'center']:
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if tokens[i-2:i+1] == ['in', 'the', 'centre']:
            tags[i-2] = 'area=city centre'
            tags[i-1] = 'area=city centre'
            tags[i] = 'area=city centre'
        if t == 'riverside':
            tags[i] = 'area=riverside'
        if t == 'waterfront':
            tags[i] = 'area=riverside'
        if t == 'river':
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['near', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['along', 'the', 'water']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['by', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['at', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-3:i+1] == ['close', 'to', 'the', 'river']:
            tags[i-3] = 'area=riverside'
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['along', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['on', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['off', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'
        if tokens[i-2:i+1] == ['beside', 'the', 'river']:
            tags[i-2] = 'area=riverside'
            tags[i-1] = 'area=riverside'
            tags[i] = 'area=riverside'


        apply_rule(i, tokens, tags, 
                   ['mid', '-', 'range'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['priced', 'in', 'the', 'higher', 'range',],
                   'price_range=high')

        apply_rule(i, tokens, tags,
                   ['customer', 'satisfaction', 'is', 'average'],
                   'customer_rating=average')
        #### family_friendly=yes

        if t == 'family':
            tags[i] = 'family_friendly=yes'

        for x1 in ['kids', 'children', 'families']:
            if tokens[i-3:i+1] == ['recommended', 'to', 'take'] + [x1]:
                for j in range(i-3,i+1):
                    tags[j] = 'family_friendly=yes'

        if tokens[i-1:i+1] == ['for', 'kids']:
            tags[i] = 'family_friendly=yes'
        if tokens[i-1:i+1] == ['for', 'children']:
            tags[i] = 'family_friendly=yes'
        if tokens[i-1:i+1] == ['for', 'children']:
            tags[i] = 'family_friendly=yes'
        if tokens[i-1:i+1] == ['for', 'family']:
            tags[i] = 'family_friendly=yes'
        if tokens[i-1:i+1] == ['for', 'families']:
            tags[i] = 'family_friendly=yes'
        if tokens[i-1:i+1] == ['family', 'place']:
            tags[i] = 'family_friendly=yes'
            tags[i-1] = 'family_friendly=yes'
        if tokens[i-5:i+1] == ['place', 'to', 'bring', 'the', 'whole', 'family']:
            for j in range(i-5,i+1):
                tags[j] = 'family_friendly=yes'

        if tokens[i-4:i+1] == ['place', 'to', 'bring', 'the', 'family']:
            for j in range(i-4,i+1):
                tags[j] = 'family_friendly=yes'

        for x1 in ['friendly', 'suitable', 'good', 'oriented', 'orientated', 
                    'open', 'opened']:
            for x2 in ["at", "with", "to", "for"]:
                for x3 in [[], ['the']]:
                    for x4 in [['kid'], ['kids'], ['child'], ['children'], 
                               ['family'], ['families'], ['all', "ages"], 
                               ["all", "age"]]:
                        p = [x1] + [x2] + x3 + x4
                        if tokens[i-len(p) +1:i+1] == p:
                            for j in range(i-len(p)+1,i+1):
                                tags[j] = 'family_friendly=yes'

        for x1 in ['child', 'children', 'family', 'kid', 'kids']:
            for x2 in [[], ['-'], ['are']]:
                for x3 in ['friendly', 'oriented', 'orientated']:
                    p = [x1] + x2 + [x3]
                    if tokens[i-len(p)+1:i+1] == p:
                        for j in range(i-len(p)+1,i+1):
                            tags[j] = 'family_friendly=yes'

#            '(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is) (?:welcome|allowed|accepted)',
        for x1 in ['kids', 'child', 'children', 'family', 'families']:
            for x2 in ['are', 'is']:
                for x3 in ['welcome', 'allowed', 'accepted']:
                    p = [x1, x2, x3]
                    if tokens[i-len(p)+1:i+1] == p:
                        for j in range(i-len(p)+1,i+1):
                            tags[j] = 'family_friendly=yes'
#            '(?:welcomes?|allows?|accepts?) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)',
        for x1 in ['welcomes', 'allows', 'accepts', 'welcome']:
            for sp in range(0,3):
                for x2 in [['kids'], ['kid'], ['child'], ['children'],
                           ['family'], ['families'], ['all', 'ages'], 
                           ['all' 'age']]:
                    p1 = [x1]
                    p2 = x2
                    l1 = i - len(p1) - len(p2) - sp + 1
                    l2 = l1 + len(p1)
                    l3 = l2 + sp
                    if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2:
                        for j in range(l1,i+1):
                            tags[j] = 'family_friendly=yes'
                        


        #### family_friendly=no

        for x1 in ['kids', 'children', 'families']:
            if tokens[i-4:i+1] == ['not', 'recommended', 'to', 'take'] + [x1]:
                for j in range(i-4,i+1):
                    tags[j] = 'family_friendly=no'


        if t == 'adult':
            tags[i] = 'family_friendly=no'       
        if tokens[i-1:i+1] == ['adults', 'only']:
            tags[i] = 'family_friendly=no'       
            tags[i-1] = 'family_friendly=no'       
        if tokens[i-2:i+1] == ['adults', '-', 'only']:
            tags[i] = 'family_friendly=no'       
            tags[i-1] = 'family_friendly=no'       
            tags[i-2] = 'family_friendly=no'       
        if tokens[i-1:i+1] == ['adult', 'only']:
            tags[i] = 'family_friendly=no'       
            tags[i-1] = 'family_friendly=no'       
        if tokens[i-1:i+1] == ['for', 'adults']:
            tags[i] = 'family_friendly=no'       
 
        #'(?:isn\'t|not|non|no)[ -]+(?:\w+ ){0,2}(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)',
        for x1 in ["n't", 'not', 'non', 'no', 'none']:
            for x2 in [[], ['-']]:
                for sp in range(0,3):
                    for x3 in ['child', 'children', 'family', 'kids', 'kid']:
                        for x4 in [[], ['-']]:
                            for x5 in ['friendly', 'oriented', 'orientated']:
                                p1 = [x1] + x2
                                p2 = [x3] + x4 + [x5]
                                l1 = max(i -len(p1) -len(p2) -sp   + 1 ,0)
                                l2 = l1 + len(p1)
                                l3 = l2 + sp
                                if tokens[l1:l2] == p1 and \
                                        tokens[l3:i+1] == p2:
                                    for j in range(l1, i+1):
                                        tags[j] = 'family_friendly=no'

#       '(?:child|children|family|kids|kid)[ -]+unfriendly',
        for x1 in ['child','children', 'family', 'kids', 'kid']:
            for x2 in [[], ['-']]:
                for x3 in [['unfriendly'], ['free']]:
                    p = [x1] + x2 + x3
                    if tokens[i-len(p)+1:i+1] == p:
                        for j in range(i-len(p)+1,i+1):
                            tags[j] = 'family_friendly=no'

        # '(?:no|not) (?:kids|children|famil(?:y|ies))',
        for x1 in ['no', 'not', "n't"]:
            for z in [[], ['place', 'for']]:
                for x2 in ['kids', 'children', 'family', 'famlies']:
                    p = [x1] + z + [x2]
                    s = max(i - len(p) + 1, 0)
                    if tokens[s:i+1] == p:
                        for j in range(s, i+1):
                            tags[j] = 'family_friendly=no'
        for x1 in ['no', 'not']:
            for x2 in ['kids', 'children', 'family', 'famlies']:
                for x3 in ['is', 'are']:
                    for x4 in ['allowed', 'welcome']:
                        if tokens[i-3:i+1] == [x1, x2, x3, x4]: 
                            tags[i] = 'family_friendly=no'
                            tags[i-1] = 'family_friendly=no'
                            tags[i-2] = 'family_friendly=no'
                            tags[i-3] = 'family_friendly=no'



#            '(?:does not|doesn\'t) (?:welcome|allow|accept) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)',
        for x1 in [['not'], ["n't"], ["no"], ["n't", "even"]]:
            for x2 in ['welcome', 'allow', 'accept', 'for']:
                for sp in range(0,2):
                    for x3 in [['kid'], ['kids'], ['child'], ['children'], 
                               ['family'], ['families'], ['all', 'ages'],
                               ['all', 'age']]:
                        p1 = x1 + [x2]
                        p2 = x3
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp 
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2:
                            for j in range(l1, i+1):
                                tags[j] = 'family_friendly=no'
                
#            '(?:not|no)(?: good| suitable| friendly| orien(?:ta)?ted| open(?:ed))? (?:at|for|to|with)(?: the)? (?:kids|children|family|families|all age)',
        for x1 in [['not'], ['no']]:
            for x2 in [['good',], ['suitable'], ['friendly'], ['oriented'],
                       ['orientated'], ['open'], ['opened']]:
                for x3 in [['at'], ['for'], ['to'], ['with']]:
                    for x4 in [[], ['the']]:
    
                        for x5 in [['kid'], ['kids'], ['child'], ['children'], 
                                   ['family'], ['families'], ['all', 'ages'],
                                   ['all', 'age']]:
                            p = x1 + x2 + x3 + x4 + x5
                            s = max(i - len(p) + 1, 0)
                            if tokens[s:i+1] == p:
                                for j in range(s, i+1):
                                    tags[j] = 'family_friendly=no'
        for x0 in ['non', 'no']:
            for x1 in ['kids', 'families', 'children', 'kid', 'family', 'child']:
                p = [x0, x1]
                if tokens[i-len(p)+1: i+1] == p:
                    for j in range(i-len(p)+1, i+1):
                        tags[j] = 'family_friendly=no'

#            '(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is)(?:n\'t| not) (?:welcome|allowed|accepted)',
        for x1 in ['kids', 'families', 'children', 'kid', 'family', 'child']:
            for x2 in [['are'], ['is'], []]:
                for x3 in ["n't", 'not']:
                    for x4 in ['welcome', 'allowed', 'accepted']:
                        p = [x1] + x2 + [x3, x4]
                        s = max(i - len(p)+ 1, 0)
                        if tokens[s:i+1] == p:
                            for j in range(s, i+1):
                                tags[j] = 'family_friendly=no'
                        

#            'adult (?:establishment|venue|place|establish)',
#        ],
#


        #### customer_rating=low
        for x0 in [[], ['average', 'customer']]:
            for x1 in ['rating', 'ratings', 'rating', 'rate', 'rated', 
                       'rates', 'review', 'reviews', 'standard', 'standards',
                       'quality']:
                for sp in range(0,3):
                    for x2 in [[], ['as']]:
                        for x3 in ['low', 'bad', 'poor', 'poorly']:
                            p1 = x0 + [x1]
                            p2 = x2 + [x3]
                            l1 = max(i - len(p1) - len(p2) - sp +1, 0)
                            l2 = l1 + len(p1)
                            l3 = l2 + sp

                            if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                    and 'but' not in tokens[l2:l3] \
                                    and ',' not in tokens[l2:l3] \
                                    and 'at' not in tokens[l2:l3] \
                                    and 'in' not in tokens[l2:l3] \
                                    and 'with' not in tokens[l2:l3] \
                                    and 'and' not in tokens[l2:l3]:
                                for j in range(l1,i+1):
                                    tags[j] = 'customer_rating=low'
       
        for x1 in ['low', 'lowly', 'bad', 'badly', 'poor', 'poorly',
                   'lower']:
            for x2 in [[], ['-'], ['in'],]:
                for x3 in [[], ['customer'], ['costumer'], ['customers'], 
                           ['consumer'], ['satisfaction'],
                           ['customer', 'service'], ['approval'],['client']]:
                    for x4 in [[], ['-']]:
                        for x5 in ['ratings', 'rated', 'rating', 'review', 
                                   'reviewed',
                                   'reviews', 'rank', 'satisfaction', 
                                   'ranking',
                                   'standard', 'standards', 'quality', 'range']:
                            p = [x1] + x2 + x3 + x4 + [x5]
                            if tokens[i-len(p)+1:i+1] == p:
                                for j in range(i-len(p) + 1, i+1):
                                    tags[j] = 'customer_rating=low'

#            "(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (?:well|good))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
#        ],


        #### customer_rating=average
        for x1 in ['rating', 'ratings', 'rating', 'rate', 'rated', 'rates',
                   'review', 'reviews', 'standard', 'standards', 'quality']:
            for sp in range(0,3):
                for x2 in [[], ['as']]:
                    for x3 in ['average', 'okay', 'ok']:
                        p1 = [x1]
                        p2 = x2 + [x3]
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and 'with' not in tokens[l2:l3] \
                                and 'and' not in tokens[l2:l3] \
                                and 'place' not in tokens[l2:l3] \
                                and 'but' not in tokens[l2:l3] \
                                and 'in' not in tokens[l2:l3] \
                                and 'kids' not in tokens[l2:l3] \
                                and '.' not in tokens[l2:l3] \
                                and 'fast' not in tokens[l2:l3] \
                                and ',' not in tokens[l2:l3]:

                            for j in range(l1,i+1):
                                tags[j] = 'customer_rating=average'
 
        if tokens[i-2:i+1] == ['average', 'english', 'food']:
            tags[i-2] = 'customer_rating=average'
        if tokens[i-2:i+1] == ['average', 'english', 'cuisine']:
            tags[i-2] = 'customer_rating=average'
        if tokens[i-1:i+1] == ['average', 'food']:
            tags[i-1] = 'customer_rating=average'
        if tokens[i-2:i+1] == ['average', 'range', 'food']:
            tags[i-2] = 'customer_rating=average'
        if tokens[i-3:i+1] == ["n't", 'take', 'your', 'family']:
            for j in range(i-3,i+1):
                tags[j] = 'family_friendly=no'

        for x1 in [['average'], ['averagely'], ['ok',], ['okay'], ['good'],
                    ['moderately'],['moderate'], ['decent']]:
            for sp in range(0,3):
                for x2 in [[], ['-']]:
                    for x3 in ['rated', 'ratings', 'rates', 'rating', 
                               'review', 'reviews', 'standard', 'standards',
                               'quality']:
                        p1 = x1
                        p2 = x2 + [x3]
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and 'and' not in tokens[l2:l3] \
                                and 'but' not in tokens[l2:l3] \
                                and 'with' not in tokens[l2:l3] \
                                and 'priced' not in tokens[l2:l3] \
                                and 'prices' not in tokens[l2:l3] \
                                and ',' not in tokens[l2:l3]:
                            for j in range(l1,i+1):
                                tags[j] = 'customer_rating=average'
 
                    




        #### customer_rating=high

        for x1 in ['rating', 'ratings', 'rating', 'rate', 'rated', 'rates',
                   'review', 'reviews', 'standard', 'standards', 'quality']:
            for sp in range(0,3):
                for x2 in [[], ['as']]:
                    for x3 in ['high', 'highly', 'excellent']:
                        p1 = [x1]
                        p2 = x2 + [x3]
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and ',' not in tokens[l2:l3] \
                                and 'at' not in tokens[l2:l3] \
                                and '.' not in tokens[l2:l3] \
                                and 'in' not in tokens[l2:l3] \
                                and 'given' not in tokens[l2:l3] \
                                and 'but' not in tokens[l2:l3] \
                                and 'is' not in tokens[l2:l3] \
                                and 'priced' not in tokens[l2:l3] \
                                and 'for' not in tokens[l2:l3] \
                                and 'that' not in tokens[l2:l3] \
                                and 'with' not in tokens[l2:l3] \
                                and 'and' not in tokens[l2:l3]:
                            for j in range(l1,i+1):
                                if not tags[j].startswith('name'):
                                    tags[j] = 'customer_rating=high'
 
        for x1 in [['high'], ['highly'], ['excellent'], ['excellently'],
                   ['very', 'good'], ['really', 'good'], ['higher'],
                   ['great'], ['greatly'], ['well'], ['outstanding'],
                   ['exceptional']]:
            for sp in range(0,3):
                for x2 in [[], ['-']]:
                    for x3 in ['rated', 'ratings', 'rates', 'rating', 
                               'review', 'reviews', 'standard', 'standards',
                               'quality', 'service', 'marks', 'rate',
                               'acclaimed', 'recommended', 'customer',
                               'regarded']:
                        p1 = x1
                        p2 = x2 + [x3]
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = max(l1 + len(p1), 0)
                        l3 = max(l2 + sp, 0)
                        
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and 'and' not in tokens[l2:l3] \
                                and 'ad' not in tokens[l2:l3] \
                                and 'while' not in tokens[l2:l3] \
                                and '.' not in tokens[l2:l3] \
                                and 'priced' not in tokens[l2:l3] \
                                and 'prices' not in tokens[l2:l3] \
                                and 'NEAR' not in tokens[l2:l3] \
                                and ',' not in tokens[l2:l3] \
                                and 'with' not in tokens[l2:l3] \
                                and 'which' not in tokens[l2:l3] \
                                and 'but' not in tokens[l2:l3]:
                            for j in range(l1,i+1):
                                tags[j] = 'customer_rating=high'
 
                    

        #### price_range=cheap
        if t in ['inexpensive', 'inexpensively', 'cheap', 'cheaply',
                 'affordable', 'affordably']:
            tags[i] = 'price_range=cheap'
        if tokens[i-3:i+1] == ['prices', 'are', 'a', 'steal']:
            for j in range(i-3, i+1):
                tags[j] = 'price_range=cheap'
        for x1 in [['low'], ['lower']]:
            for x2 in [[], ['-']]:
                for x3 in ['price', 'priced', 'prices', 'cost']:
                    p = x1 + x2 + [x3]
                    s = max(i - len(p) + 1, 0)
                    if tokens[s:i+1] == p:
                        for j in range(s,i+1):
                            tags[j] = 'price_range=cheap'
        #"prices?(?: range)?(?: \w+){0,3} low",
        for x1 in [['price'], ['prices']]:
            for x2 in [[], ['range']]:
                for sp in range(0,4):
                    for x3 in [['low']]:
                        p1 = x1 + x2
                        p2 = x3
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and 'but' not in tokens[l2:l3] \
                                and 'and' not in tokens[l2:l3] \
                                and 'although' not in tokens[l2:l3] \
                                and ',' not in tokens[l2:l3] \
                                and '.' not in tokens[l2:l3] \
                                and 'whit' not in tokens[l2:l3] \
                                and 'with' not in tokens[l2:l3]:
                            for j in range(l1, i+1):
                                tags[j] = 'price_range=cheap'


        #### price_range=high
        #"high[- ]+price[ds]?",
        for x1 in [['high'], ['upper'], ['higher'], ['highly'], ['hight'],]:
            for x2 in [[], ['-'], ['in']]:
                for x3 in [['end'], ['end', 'of', 'the'], []]:
                    for x4 in ['price', 'priced', 'prices', 'costs', 'pricing',
                                'prince']:
                        p = x1 + x2 + x3 + [x4]
                        s = max(i - len(p) + 1, 0)
                        if tokens[s:i+1] == p:
                            for j in range(s,i+1):
                                tags[j] = 'price_range=high'
        if tokens[i-1:i+1] == ['high', 'end']:
            tags[i] = 'price_range=high'
            tags[i-1] = 'price_range=high'
        if tokens[i-2:i+1] == ['high', '-', 'end']:
            tags[i] = 'price_range=high'
            tags[i-1] = 'price_range=high'
            tags[i-2] = 'price_range=high'

        if t == "expensive":
            tags[i] = 'price_range=high'

        #   "prices?(?: range)?(?: \w+){0,3} high",
        for x1 in [['price'], ['prices'], ['priced'],]:
            for x2 in [[], ['range']]:
                for sp in range(0,4):
                    for x3 in [['high'], ['being', 'on', 'the', 'high', 'side']]:
                        p1 = x1 + x2
                        p2 = x3
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and 'but' not in tokens[l2:l3] \
                                and 'while' not in tokens[l2:l3] \
                                and 'and' not in tokens[l2:l3] \
                                and 'the' not in tokens[l2:l3] \
                                and 'has' not in tokens[l2:l3] \
                                and 'that' not in tokens[l2:l3] \
                                and 'a' not in tokens[l2:l3] \
                                and ',' not in tokens[l2:l3] \
                                and 'with' not in tokens[l2:l3]:
                            for j in range(l1, i+1):
                                tags[j] = 'price_range=high'




        #### price_range=less than 20
        for x0 in [[], 
                   ['the', 'low', 'price', 'range', 'of'], 
                   ['low', 'price', 'range', 'of', 'items'], 
                   ['price', 'range', 'is', 'low', 'being'],
                   ['good', 'price', 'range', ','],
                   ['affordable', 'price', 'range', 'at',],
                   ['low', 'cost', 'price', 'range',],
                   ['well', 'priced', 'at'], 
                   ['moderate', 'price', 'range', 'of'],
                   ['low', 'price', 'range', ',',],
                   ['affordable', 'costs'],
                   ['cheap', 'food', 'outlet', 'no',],
                   ['average', 'price', 'range', 'of'], 
                   ['low', 'price', 'range', 'of',],
                   [ 'average', 'price', 'range', 'is'],
                   [ 'average', 'price', 'is'],
                   ['cheap','food', 'at', ],
                   ['cheap',',']]:
            for x1 in [['less', 'than'], ['under'], 
                       ['less', 'that'],
                       ['less'],
                       ['no', 'more', 'that'],
                       ['less', 'then'],
                       ['lower', 'than'], ['below'],]:
                for x2 in [['£', '20'], ['20']]:
                    for x3 in [
                        [',', 'it', 'is', 'low', 'priced'], ['english', 'sterling'], []]:
                        p = x0 + x1 + x2 + x3
                        s = max(i - len(p) + 1, 0)
                        if tokens[s:i+1] == p:
                            for j in range(s, i+1):
                                tags[j] = 'price_range=less than £20'

#        if tokens[i-3:i+1] == ['less', 'than', '£', '20']:
#            for j in range(i-3,i+1):
#                tags[j] = 'price_range=less than £20'
#        if tokens[i-2:i+1] == ['less', 'than', '20']:
#            for j in range(i-2,i+1):
#                tags[j] = 'price_range=less than £20'
#        if tokens[i-2:i+1] == ['under', '£', '20']:
#            for j in range(i-2,i+1):
#                tags[j] = 'price_range=less than £20'
#        if tokens[i-1:i+1] == ['under', '20']:
#            for j in range(i-1,i+1):
#                tags[j] = 'price_range=less than £20'
#
        #### price_range=more than 30

        if tokens[i-4:i+1] == ['30', 'and', 'higher', 'price', 'range']:
            for j in range(i-4,i+1):
                tags[j] = 'price_range=more than £30'

        for x0 in [['average', 'price', 'range', 'of'], 
                   ['average', 'prices', 'are'],
                   ['average', 'price', 'is'],
                   ['expensive', 'at',],
                   ['average', 'price'],
                   ['prices', 'are', 'high', 'at',],
                   ['high', 'price', 'range', 'expected', 'to', 'be'],
                   ['high', 'price', 'range', 'of'],
                   ['prices', 'average',],
                   ['higher', 'price'],
                   ['average', 'price', 'for', 'a', 'meal', 'is'],
                   ['to', 'expensive', 'since', 'it', 'cost'],
                   ['expensive', ',', 'usually', 'charging'], 
                   ['its', 'price', 'range', 'is', 'quite', 'high', ','],
                   ['above', 'average', 'price', 'range'],
                   ['average', 'price', 'range', 'is'],
                   ['on', 'average', 'costing'], 
                   ['higher', 'price', 'range', 'of',], []]:
            for x1 in [['more', 'than'], ['over'], ['priced', 'at'], 
                       ['more', 'then'], 
                       ['more', 'that'], 
                       ['more', 'then', 'the'], 
                       ['greater', 'that',], ['above'], ['above', 'the'],[]]:
                for x2 in [['£', '30'], ['30']]:
                    for x3 in [['english', 'sterling'], ['british', 'pounds'],
                               []]:
                        p = x0 + x1 + x2 + x3
                        s = i - len(p) + 1
                        if tokens[s:i+1] == p:
                            for j in range(s, i+1):
                                tags[j] = 'price_range=more than £30'

        #### price_range=£20-25
        for x0 in [['average', 'price', 'range', 'of',], 
                   ['average', 'prices', 'between',], 
                   ['affordable', 'rate', 'of'],
                   ['average', 'cost', 'of'],
                   ['reasonable', 'price', 'range', 'of'],
                   ['average', 'prices',], 
                   ['price', 'range', 'is', 'average', 'at',],
                   ['moderately', 'priced', 'at'],
                   ['average', 'price', 'is'], 
                   ['average', 'price', 'of',], 
                   ['priced', 'at', 'an', 'average'],
                   ['less', 'than'],
                   ['low', 'price', 'range', 'of'],
                   ['average', 'price', 'range', 'of', 'around',],
                   ['averagely', 'priced', 'at'],
                   ['average', 'prices', 'range', 'between'],
                   ['average', 'prices', 'of',], 
                   ['reasonably', 'priced', 'with', 'meals', 'costing', 'around'],
                   ['relatively', 'cheap', 'with', 'the', 'prices', 'being', 'around'],
                   ['reasonable', 'prices', 'of'],
                   ['medium', 'price', 'range', 'fare'],
                   ['moderately', 'expensive', ',', 'at'],
                   ['average', 'price', 'being',],
                   ['average', 'price', 'range','is',], 
                   ['affordable', 'price', 'range', 'of',], 
                   ['affordable', 'price', 'range', 'between',], 
                   ['high', 'price', 'range'],
                   ["n't", 'cheap', ',', 'its', 'between'],
                   ['low', 'prices', 'starting', 'at'  ], []]:
            for x1 in [['£'], ['e'], []]:
                for x2 in [['20']]:
                    for x3 in [['-'], ['--'], ['–'], ['––'], ['to'], ['and']]:
                        for x4 in [['£'], ['e'], []]:
                            for x5 in [['25'], ['30'],]:
                                for x6 in [['english', 'sterling'], 
                                           ['pounds'],
                                           [',', 'which', 'is', 'the', 
                                            'average', 'price', 'range'],
                                           []]:
                                    p = x0 + x1 + x2 + x3 + x4 + x5 + x6
                                    s = i - len(p) + 1
                                    if tokens[s:i+1] == p:
                                        for j in range(s, i+1):
                                            tags[j] = 'price_range=£20-25'

        #### price_range=moderate
        for x1 in ['moderate', 'moderately', 'reasonable', 'reasonably',
                   'ok', 'average', 'averagely', 'mid', 'medium', 'well',
                    'middle', 'good', 'decent', 'modest', 'middling']:
            for x2 in [[], ['-']]:
                for x3 in [['level'], []]:
                    for x4 in [['price'], ['prices'], ['priced'], ['pricing'],
                               ['value'], ['food', 'bracket']]:
                        p = [x1] + x2 + x3 + x4
                        s = max(i-len(p)+1, 0)
                        if tokens[s:i+1] == p:
                            for j in range(s, i+1):
                                tags[j] = 'price_range=moderate'
        if tokens[i-1:i+1] == ['not', 'cheap']:
            for j in range(i-1,i+1):
                tags[j] = 'price_range=high'
        for x1 in ['mid']:
            for x2 in [[], ["-"]]:
                for x3 in ['range']:
                    for x4 in [[], ['-']]:
                        for x5 in ['price', 'prices', 'priced']:
                            p = [x1]+x2 + [x3] + x4 + [x5]
                            s = max(i - len(p) + 1, 0)
                            if tokens[s:i+1] == p:
                                for j in range(s, i+1):
                                    tags[j] = 'price_range=moderate'
        #"prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",
        for x1 in ['price', 'prices', 'priced']:
            for x2 in [[], ['range'], ['range', 'is', 'from'],]:
                for sp in range(0, 3):
                    for x3 in ['ok', 'average', 'modestly', 'moderate', 'moderately',
                               'reasonable']:
                        p1 = [x1] + x2
                        p2 = [x3]
                        l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                        l2 = l1 + len(p1)
                        l3 = l2 + sp
                        if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2 \
                                and 'with' not in tokens[l2:l3] \
                                and ',' not in tokens[l2:l3] \
                                and '.' not in tokens[l2:l3] \
                                and 'chinese' not in tokens[l2:l3] \
                                and 'however' not in tokens[l2:l3] \
                                and 'that' not in tokens[l2:l3] \
                                and 'yet' not in tokens[l2:l3] \
                                and 'no' not in tokens[l2:l3] \
                                and 'but' not in tokens[l2:l3] \
                                and 'than' not in tokens[l2:l3] \
                                and 'and' not in tokens[l2:l3]:
                            for j in range(l1,i+1):
                                tags[j] = 'price_range=moderate'

        #### customer_rating=5 out of 5
        for x0 in [['high', 'level', 'of', 'service', 'with', 'an', 'average',
                    'customer', 'rating', 'of',],
                   ['average', 'rating', 'of'], 
                   []]:
            for x1 in [['5'], ['five']]:
                for x2 in [[], ['-']]:
                    for x3 in [['star'], ['stars']]:
                        p = x0 + x1 + x2 + x3
                        s = i - len(p) + 1
                        if tokens[s:i+1] == p:
                            for j in range(s,i+1):
                                tags[j] = 'customer_rating=5 out of 5'


        for x0 in [['rated',], ['rated', 'at']]:
            for x1 in [['5'], ['five']]:
                p = x0 + x1
                s = i - len(p) + 1
                if tokens[s:i+1] == p:
                    for j in range(s, i+1): 
                        tags[j] = 'customer_rating=5 out of 5'

        #### customer_rating=3 out of 5
        for x0 in [['a', 'great', 'customer', 'rating', 'of'],   
                   ['average', 'customer', 'rating', 'of'],
                   ['average', 'rated'],
                   ['average', 'customer', 'rating', ':'],
                   ['rating', 'is', 'an', 'average'],
                   ['average', 'reviews', 'of',],
                   ['decent', 'rating', 'of'],
                   ['average', 'customer', 'rating', 'is'],
                   ['average', 'customer', 'rating',],
                   ['average', 'ratings', 'a',],
                   ['average', 'rating', 'is'],
                   ['average', 'rating', 'of'], ['well', 'rated'],
                   []]:
            for x1 in ['3', 'three']:
                for x2 in [['out', 'of',], ['of'], ['-'],]:
                    for x3 in [['five'], ['5']]:
                        for x4 in [[], ['star'], ['stars']]:
                            for x5 in [
                                    [],
                                    ['customer', 'ratings', 'and', 
                                     'excellent', 'service']]:
                                p = x0 + [x1] + x2 + x3 + x4 + x5
                                s = i - len(p) + 1
                                if tokens[s:i+1] == p:

                                    for j in range(s,i+1):
                                        tags[j] = 'customer_rating=3 out of 5'
        for x1 in [['3'], ['three']]:
            for x2 in [[], ['-']]:
                for x3 in [['star'], ['stars']]:
                    p = x1 + x2 + x3
                    s = i - len(p) + 1
                    if tokens[s:i+1] == p:
                        for j in range(s,i+1):
                            tags[j] = 'customer_rating=3 out of 5'

               
        #### customer_rating=1 out of 5
        for x0 in [['rating', 'is', 'a', 'low'],
                   ['poor', 'customer', 'rating', ','], 
                   ['poor', 'customer', 'rating', 'of'], 
                   ['lower', 'customer', 'service', 'rating', 'of'],
                   ['poor', 'rating', 'of'],
                   ['a', 'high', 'customer', 'rating', 'of',],
                   ['low', 'customer', 'rating'],
                   ['low', 'customer', 'rating', 'of'], 
                   ['low', 'rated', 'as'], 
                   ['low', 'rating', 'of'],
                   ['ratings', 'are', 'a', 'low',],
                   ['moderate', 'customer', 'rating', 'of'],
                   ['average', 'customer', 'rating',], 
                   ['average', 'customer', 'rating', 'is',], 
                   ['average', 'customer', 'rating', 'is', 'only'], 
                   ['average', 'customer', 'rating','of',], 
                   ['average', 'rating','of',], 
                   ['a', 'low', 'customer', 'rating', 'of',], []]:
            for x1 in ['1', 'one']:
                for x2 in [['out', 'of',], ['of'], ['out'], ['-'],]:
                    for x3 in [['five'], ['5']]:
                        for x4 in [['star'], ['stars'], 
                                   ['average', 'customer', 'rating'], []]:
                            p = x0 + [x1] + x2 + x3 + x4
                            s = i - len(p) + 1
                            if tokens[s:i+1] == p:
                                for j in range(s,i+1):
                                    tags[j] = 'customer_rating=1 out of 5'
        for x0 in [[], ['low', 'customer', 'rating', 'of', ], ['rated', 'a', 'low'],]:
            for x1 in [['1'], ['one']]:
                for x2 in [[], ['-']]:
                    for x3 in [['star'], ['stars'], ['start']]:
                        p = x0 + x1 + x2 + x3
                        s = max(i - len(p) + 1, 0)
                        if tokens[s:i+1] == p:
                            for j in range(s,i+1):
                                tags[j] = 'customer_rating=1 out of 5'

        #### customer_rating=5 out of 5
        for x0 in [['a', 'high', 'rating', 'of'], 
                   ['a', 'high', 'rating', 'from', 'its', 'customers', 'of'], 
                   ['high', 'customer', 'rating', 'as'], 
                   ['high', 'customer', 'ratings', 'of',],
                   ['rate', 'it', 'highly', 'at',],
                   ['excellent', 'ratings', 'of'],
                   ['excellent', 'customer', 'review', 'of',],
                   ['outstanding', 'customer', 'rating', 'of'],
                   ['great', 'restaurant', 'rated'],
                   ['high', 'ratings', 'of'],
                   ['high', 'customer', 'rating', 'of'], 
                   ['good', 'reviews', ',', 'with', 'a'],
                   ['highly', 'rated', ','],
                   ['highly', 'rated', 'with'], 
                   ['average', 'customer', 'rating', 'of'],
                   ['average', 'rating', 'of'],
                   ['an', 'excellent', 'customer', 'rating', 'of'], 
                   ['a', 'great', 'customer', 'rating', 'of',], []]:
            for x1 in ['5', 'five']:
                for x2 in [['out', 'of',], ['of'], ['out']]:
                    for x3 in [['five'], ['5']]:
                        p = x0 + [x1] + x2 + x3
                        s = i - len(p) + 1
                        if tokens[s:i+1] == p:
                            for j in range(s,i+1):
                            
                                tags[j] = 'customer_rating=5 out of 5'
     

        if tokens[i-7:i+1] == ['prices', 'as', 'well', 'as', 'the', 
                               'reviews', 
                               'are', 'decent']:
            tags[i-7] = 'price_range=moderate'
            for j in range(i-6,i+1):
                tags[j] = 'customer_rating=average'
 
        if tokens[i-3:i+1] == ['average', 'rated', 'low', 'cost']:
            tags[i-3] = 'customer_rating=average'
            tags[i-2] = 'customer_rating=average'
            tags[i-1] = 'price_range=cheap'
            tags[i] = 'price_range=cheap'
        if tokens[i-3:i+1] == ["n't", 'break', 'your', 'budget']:
            for j in range(i-3, i+1):   
                tags[j] = 'price_range=cheap'
        if tokens[i-5:i+1] == ['for', 'the', 'restaurant', 'the', 'rice', 
                               'boat']:
            tags[i] = 'near=The Rice Boat'
            tags[i-1] = 'near=The Rice Boat'
        if tokens[i-5:i+1] == ['not', 'give', 'it', 'a', 'high', 'rating']:
            for j in range(i-5,i+1):
                tags[j] = 'customer_rating=low'

        if t == 'café' and tags[i] == '0':
            tags[i] = 'eat_type=coffee shop'
        for x0 in ['good', 'average']:
            for x1 in ['french', 'italian', 'chinese', 'indian','japanese',
                       'english', 'british']:
                for x2 in ['food', 'eats']:
                    p = [x0, x1, x2]
                    s = max(i - len(p) + 1, 0)
                    if tokens[s:i+1] == p:
                        tags[s] = 'customer_rating=average'
        for x1 in ['families', 'kids', 'children', 'family']:
            p = ['not', 'for', x1]
            s = i - len(p) + 1
            if tokens[s:i+1] == p:
                for j in range(s,i+1):
                    tags[j] = 'family_friendly=no'

            p = ['non', '-', 'friendly', x1]
            s = i - len(p) + 1
            if tokens[s:i+1] == p:
                for j in range(s,i+1):
                    tags[j] = 'family_friendly=no'
   

        if tokens[i-3:i+1] == ["n't", 'love', 'the', 'family']:
            for j in range(i-3,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-2:i+1] == ["aimed", "at", 'older']:
            for j in range(i-2,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-2:i+1] == ['well', 'priced', 'quality']:
            tags[i-2] = 'price_range=moderate'
            tags[i-1] = 'price_range=moderate'
            tags[i] = 'customer_rating=average'


        
        if tokens[i-1:i+1] == ['not', 'glowing']:
            tags[i] = 'customer_rating=low'
            tags[i-1] = 'customer_rating=low'
        if tokens[i-4:i+1] == ['for', 'adults', 'not', 'the', 'family']:
            for j in range(i-4,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-5:i+1] == ['not', 'a', 'good', 'place', 'for', 'kids']:
            for j in range(i-5,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-6:i+1] == ['near', 'the', 'river', 'and', 'the', 'rice',
                               'boat']:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'

        if tokens[i-5:i+1] == ['good', 'food', ',', 'and', 'customer', 
                               'satisfaction']:
            for j in range(i-5,i+1):
                tags[j] = 'customer_rating=average'
        if tokens[i-2:i+1] == ['heart', 'of', 'town']:
            for j in range(i-2,i+1):
                tags[j] = 'area=city centre'
        if tokens[i-4:i+1] == ['prices', 'that', 'are', 'very', 'reasonable']:
            for j in range(i-4,i+1):
                tags[j] = 'price_range=moderate'
        if tokens[i-2:i+1] == ['great', 'coffee', 'shop']:
            tags[i-2] = 'customer_rating=high'
        if tokens[i-1:i+1] == ['centre', 'city']:
            for j in range(i-1, i+1):
                tags[j] = 'area=city centre'
        if tokens[i-3:i+1] == ['high', 'opinion', 'of', 'customers']:
            for j in range(i-3, i+1):
                tags[j] = 'customer_rating=high'
        if tokens[i-5:i+1] == ['not', 'cheap', 'but', 'not', 'too', 'expensive']:
            for j in range(i-5,i+1):
                tags[j] = 'price_range=moderate'

        if tokens[i-3:i+1] == ['lower', 'prices', 'than', 'average',]:
            for j in range(i-3,i+1):
                tags[j] = 'price_range=cheap'
        if tokens[i-2:i+1] == ['crown', 'plaza', 'hotel']:
            for j in range(i-2,i+1):
                tags[j] = 'near=Crowne Plaza Hotel'
        if tokens[i-7:i+1] == ['average', 'customer', 'rating', 'for', 'this',
                               'restaurant', 'is', 'low']:
            for j in range(i-7,i+1):
                tags[j] = 'customer_rating=low'
        if tokens[i-2:i+1] == ['average', 'customer', 'reasons']:
            for j in range(i-2,i+1):
                tags[j] = 'customer_rating=average'
        if tokens[i-3:i+1] == ['low', 'prices', 'and', 'ratings']:
            tags[i] = 'customer_rating=low'
        if tokens[i-5:i+1] == ['not', 'recommended', 'to', 'take', 'your', 
                               'family']:
            for j in range(i-5,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-3:i+1] == ["n't", 'get', 'good', 'reviews']:
            for j in range(i-3,i+1):
                tags[j] = 'customer_rating=low'
        if tokens[i-6:i+1] == ['five', 'star', 'rating', 'with', 'good', 
                               'quality', 'food']:
            for j in range(i-6,i+1):
                tags[j] = 'customer_rating=5 out of 5'
        if tokens[i-7:i+1] == ['5', 'by', 'customers', 'as', 'it', 
                               'provides', 'great', 'service']:
            for j in range(i-7,i+1):
                tags[j] = '0'
        if tokens[i-3:i+1] == ['quality', 'is', 'not', 'abysmal']:
            for j in range(i-3,i+1):
                tags[j] = 'customer_rating=average'
        if tokens[i-1:i+1] == ['average', 'coffee']:
            tags[i-1] = 'customer_rating=average'

        if tokens[i-1:i+1] == ['hight', 'ratings']:
            tags[i-1] = 'customer_rating=high'
            tags[i] = 'customer_rating=high'
        if tokens[i-2:i+1] == ['higher', 'than', 'average']:
            for j in range(i-2,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-1:i+1] == ['for', 'city']:
            tags[i] = 'area=city centre'
        if tokens[i-3:i+1] == ['not', 'provide', 'family', 'services']:
            for j in range(i-3,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-4:i+1] == ['deliveries', 'in', 'the', 'medium', 'range',]:
            for j in range(i-4,i+1):
                tags[j] = 'price_range=moderate'
        if tokens[i-5:i+1] == ['prices', 'are', 'on', 'the', 'high', 'side']:
            for j in range(i-5,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-3:i+1] == ['age', 'minimum', 'is', '21']:
            for j in range(i-3,i+1):
                tags[j] = 'family_friendly=no'
        if tokens[i-3:i+1] == ['for', 'a', 'nice', 'price']:
            for j in range(i-3,i+1):
                tags[j] = 'price_range=moderate'
        if tokens[i-2:i+1] == ['low', 'a', 'rating']:
            for j in range(i-2,i+1):
                tags[j] = 'customer_rating=low'
        if tokens[i-2:i+1] == ['excellent', 'english', 'food']:
            tags[i-2] = 'customer_rating=high'
        if tokens[i-5:i+1] == ['reasonably', 'priced', 'at', 'about', '£', '25']:
            for j in range(i-5,i+1):
                tags[j] = 'price_range=£20-25'
        if tokens[i-4:i+1] == ['prices', 'in', 'the', 'high', 'range']:
            for j in range(i-4,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-5:i+1] == ['prices', 'are', 'in', 'the', 'high', 'range']:
            for j in range(i-5,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-3:i+1] == ['well', 'rated', 'average', 'prices']:
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'customer_rating=high'
            tags[i-1] = 'price_range=moderate'
            tags[i] = 'price_range=moderate'
        if tokens[i-4:i+1] == ['customer', 'satisfaction', 'levels', 'are', 'low']:
            for j in range(i-4,i+1):
                tags[j] = 'customer_rating=low'
        if tokens[i-3:i+1] == [ 'cheap', 'and', 'good', 'value']:
            for j in range(i-3,i+1):
                tags[j] = 'price_range=cheap'
        if tokens[i-3:i+1] == ['higher', 'than', 'average', 'prices']:
            for j in range(i-3,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-1:i+1] == ['french', 'fries']:
            tags[i-1] = '0'

        if tokens[i-4:i+1] == ['higher', 'than', 'average', 'price', 'range']:
            for j in range(i-4,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-2:i+1] == ['moderate', 'but', 'affordable']:
            for j in range(i-2,i+1):
                tags[j] = 'price_range=moderate'
        if tokens[i-1:i+1] == ['located', 'downtown']:
            tags[i] = 'area=city centre'
            tags[i-1] = 'area=city centre'
        if tokens[i-5:i+1] == ['prices', 'fall', 'in', 'the', 'moderate', 'range',]:
            for j in range(i-5,i+1):
                tags[j] = 'price_range=moderate'
        if tokens[i-2:i+1] == ['average', 'food', 'prices',]:
            for j in range(i-2,i+1):
                tags[j] = 'price_range=moderate'
        if tokens[i-8:i+1] == ['three', 'out', 'of', 'five', 'diners', 'give', 'it', 'good', 'reviews']:
            for j in range(i-8,i+1):
                tags[j] = 'customer_rating=average'
        if tokens[i-8:i+1] == ['near', 'to', 'the', 'japanese', 'restaurant', ',', 'the', 'rice', 'boat',]:
            for j in range(i-1,i+1):
                tags[j] = 'near=The Rice Boat'
        if tokens[i-4:i+1] == ['customers', 'rating', 'it', 'a', '3']:
            for j in range(i-4,i+1):
                tags[j] = 'customer_rating=3 out of 5'
        if tokens[i-3:i+1] == ['higher', 'than', 'average', 'priced']:
            for j in range(i-3,i+1):
                tags[j] = 'price_range=high'
        if tokens[i-2:i+1] == ['perfect', 'customer', 'rating']:
            for j in range(i-2,i+1):
                tags[j] = 'customer_rating=5 out of 5'
        if tokens[i-3:i+1] == ['great', 'quality', 'low', 'price']:
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'customer_rating=high'
            tags[i-1] = 'price_range=low'
            tags[i] = 'price_range=low'
        if t == 'riverfront':
            tags[i] = 'area=riverside'
        for x1 in [['price', 'range', 'of', ]]:
            for sp in range(1,3):
                for x2 in [['is', 'high']]:
                    p1 = x1
                    p2 = x2
                    l1 = max(i - len(p1) - len(p2) - sp + 1, 0)
                    l2 = l1 + len(p1)
                    l3 = l2 + sp
                    if tokens[l1:l2] == p1 and tokens[l3:i+1] == p2:
                        tags[i] = 'price_range=high'
        apply_rule(i, tokens, tags, 
                   ['leave', 'the', 'children', 'at', 'home'],
                   'family_friendly=no')


        apply_rule(i, tokens, tags, 
                   ['restaurant', 'is', 'exceptional'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['5', 'out', 'of', 'a', 'possible', '5', 'rating'],
                   'customer_rating=5 out of 5')

        apply_rule(i, tokens, tags, 
                   ['when', 'thinking', 'of', 'taking', 'the', 'children', 'out'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['on', 'the', 'pricey', 'side'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['a', 'great', 'price',],
                   'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['in', 'the', 'city', 'of', 'riverside',],
                   'area=riverside')


        apply_rule(i, tokens, tags, 
                   ['not', 'the', 'best', 'place', 'to', 'bring', 'your', 'kids'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['no', 'high', 'price', 'range'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['fair', 'priced',],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['fair', 'prices',],
                   'price_range=moderate')
        if tokens[i-3:i+1] == ['low', 'priced', 'and', 'rated']:
            tags[i-3] = 'price_range=cheap'
            tags[i-2] = 'price_range=cheap'
            tags[i] = 'customer_rating=low'

        apply_rule(i, tokens, tags, 
                   ['£', '-', '25', 'price', 'range'],
                   'price_range=£20-25')
        apply_rule(i, tokens, tags, 
                   ['hight', 'customer', 'ratings'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['good', 'food'],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['rated', '1', 'on', 'a', 'scale', 'of', '5',],
                   'customer_rating=1 out of 5')

        apply_rule(i, tokens, tags, 
                   ['travellers', 'rest', 'beefeaters',],
                   'name=Travellers Rest Beefeater')
        apply_rule(i, tokens, tags, 
                   ['leave', 'the', 'kids', 'at', 'home'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'somewhat', 'steep'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['suitable', 'for', 'all', 'families',],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ["n't", 'cater', 'well', 'to', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['not', 'likable', 'by', 'kids'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['not', 'family', 'family'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['caters', 'to', 'children'],
                   'family_friendly=yes')


        apply_rule(i, tokens, tags, 
                   ["'s", 'no', 'to', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['less', 'child', 'friendly'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['ok', 'without', 'the', 'kids'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ["n't", 'have', 'good', 'customer', 'reviews'],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['not', 'suitable', 'for', 'young', 'families'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['unsuitable', 'for', 'families'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['cost', 'effective'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['meals', 'that', 'are', 'average',],
                   'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['not', 'well', 'rated'],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['not', 'very', 'good', 'reviews'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'is', 'lower', 'than', 'average'],
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['the', 'watermans'],
                   'name=The Waterman')
        apply_rule(i, tokens, tags, 
                   ['low', '-', 'end'],
                   'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['low', 'end'],
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['not', 'too', 'cheap'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['not', 'too', 'expensive'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['quality', 'doesn', "'t", 'come', 'cheap'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'is', 'on', 'the', 'high', 'side'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'mid', '-', 'level'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   [ 'above', 'average', 'price', 'range'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['higher', 'than', 'usual', 'price', 'range'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['in', 'the', 'city'],
                   'area=city centre')
        apply_rule(i, tokens, tags, 
                   ['city', "'s", 'centre'],
                   'area=city centre')
        apply_rule(i, tokens, tags, 
                   ['downtown', 'region'],
                   'area=city centre')

        apply_rule(i, tokens, tags, 
                   ['middle', 'of', 'the', 'city'],
                   'area=city centre')

        apply_rule(i, tokens, tags, 
                   ['heart', 'of', 'the', 'city'],
                   'area=city centre')

        apply_rule(i, tokens, tags, 
                   ['high', '-', 'cost'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'that', 'is', 'rather', 'moderate'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['4.5', 'out', 'of', '5'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ["n't", 'get', 'good', 'ratings'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['4', 'out', 'of', '5'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['£', '20', 'and', 'lower', 'price'],
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'that', 'is', 'moderate'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['higher', 'end'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['bring', 'the', 'kids', 'along'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['family', "'s", 'are', 'not', 'welcome'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['not', 'a', 'place', 'for', 'a', 'family', 'outing',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['prices', 'for', 'meals', 'are', 'moderate',],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['average', 'cost'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['price', 'a', 'little', 'high'],
                   'price_range=high')

        if tokens[i-1:i+1] == ['mediocre', 'chinese']:
            tags[i-1] = 'customer_rating=low'

        apply_rule(i, tokens, tags, 
                   ['medium', 'range', 'prices'],
                   'price_range=moderate')

        if tokens[i-2:i+1] == ['city', 'centre', 'dive']:
            tags[i] = 'customer_rating=low'
        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'higher', 'then', 'average'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20'],
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['scores', 'low', 'amongst', 'customer', 'satisfaction'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['boasts', 'perfect', 'customer', 'ratings'],
                   'customer_rating=5 out of 5')

        apply_rule(i, tokens, tags, 
                   ['children', 'welcome'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['customers', 'rate', 'it', 'very', 'well'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['quality','food',],
                   'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['offers', 'average', ',',],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['food', 'is', 'amazing'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['ratings', 'are', 'not', 'high'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['welcoming', 'to', 'families'],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['caters', 'to', 'couples', 'and', 'single', 'adults'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['anti', '-', 'family'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['mainly', 'to', 'singles', 'and', 'couples'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['low', 'star', 'rating',],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['children', "'s", 'specials'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'more', 'than', 'average'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ["n't", 'bring', 'the', 'family'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['a', 'place', 'to', 'take', 'your', 'children',],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['mid', 'rated'],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['between', '3', 'and', '5'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['not', 'of', 'very', 'good', 'quality'],
                   'customer_rating=low')


        apply_rule(i, tokens, tags, 
                   ['high', 'price', '£', '30',],
                   'price_range=more than £30')

        apply_rule(i, tokens, tags, 
                   ['downtown'],
                   'area=city centre')

        apply_rule(i, tokens, tags, 
                   ['do', 'not', 'cater', 'for', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['loves', 'to', 'have', 'children',],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['childless', 'atmosphere'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['customers', 'poorly', 'rate'],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, ['outstanding'], 'customer_rating=high')
        if tokens[i-3:i+1] == ['moderately', 'rated', 'and', 'prices']:
            tags[i] = 'price_range=moderate'
        if tokens[i-3:i-1] == ['very', 'good'] and tokens[i] == 'food':
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'customer_rating=high'
        if tokens[i-2:i+1] == ['high', 'priced', 'average']:
            tags[i-1] = 'price_range=high'
        if tokens[i-7:i+1] == ["n't", 'rated', 'very', 'high', 'for', 'their',
                               'moderate', 'pricing',]:
            for j in range(i-7,i-3):
                tags[j] = 'customer_rating=average'
        if tokens[i-6:i+1] == ['vegetarian', 'café', 'is', 'near', 'the', 'rice', 'boat']:
            tags[i] = 'name=The Rice Boat'
            tags[i-1] = 'name=The Rice Boat'
        if tokens[i-4:i+1] == ['served', 'in', 'the', 'rice', 'boat']:
            tags[i] = 'name=The Rice Boat'
            tags[i-1] = 'name=The Rice Boat'
            tags[i-2] = 'name=The Rice Boat'

        if tokens[i-8:i+1] == ['you', 'can', 'get', 'english', 'food', 'in', 'the', 'rice', 'boat']:

            tags[i] = 'name=The Rice Boat'
            tags[i-1] = 'name=The Rice Boat'
            tags[i-2] = 'name=The Rice Boat'

        if tokens[i-8:i+1] == ['charges', 'a', 'bit', 'more', 'than', 'the', 
                                'average', 'coffee', 'shop']:
            for j in range(i-8,i-1):
                tags[j] = 'price_range=high'
            for j in range(i-1,i+1):
                tags[j] = 'eat_type=coffee shop'

        if tokens[i-5:i+1] == ['customers', 'rated', 'the', 'golden', 'palace', 'highly']:
            tags[i] = 'customer_rating=high'
        if tokens[i-3:i+1] == ['customers', 'rated', 'NAME', 'highly']:
            tags[i] = 'customer_rating=high'
            tags[i-3] = '0'
            tags[i-2] = '0'
        if tokens[i-4:i+1] == ['customers', 'rated', 'the', 'NAME', 'highly']:
            tags[i] = 'customer_rating=high'
            tags[i-4] = '0'
            tags[i-3] = '0'
            tags[i-2] = '0'


        if tokens[i-4:i+1] == ['high', 'priced', 'low', 'customer', 'rated',]:
            for j in range(i-4,i-2):
                tags[j] = 'price_range=high'
            for j in range(i-2,i+1):
                tags[j] = 'customer_rating=low'

        if tokens[i-4:i+1] == ['high', 'priced', 'low', 'customer', 'rating']:
            tags[i-4] = 'price_range=high'
            tags[i-3] = 'price_range=high'
            tags[i-2] = 'customer_rating=low'
            tags[i-1] = 'customer_rating=low'
            tags[i] = 'customer_rating=low'
        if tokens[i-17:i+1] == ['price', 'range', 'is', 'a', 'bit', 'higher',
                                'than', 'most', 'but', 'for','the', 'quality',
                                'of', 'food', 'it', 'is', 'worth', 'it']:
            for j in range(i-17,i-9):
                tags[j] = 'price_range=high'
            for j in range(i-8,i+1):
                tags[j] = 'customer_rating=high'
        if tokens[i-3:i+1] == ['an', 'average', 'italian', 'place']:
            for j in range(i-3,i-1):
                tags[j] = 'customer_rating=average'
        apply_rule(i, tokens, tags, ['fast', 'foods'], 'food=Fast food')
        apply_rule(i, tokens, tags, 
                   ['at', 'a', 'fair', 'price'], 'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['favourable', 'reviews'], 'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['only', 'spend', 'a', 'moderate', 'amount', 'of', 'money'], 
                    'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   [ 'not', 'family', 'place'], 'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['without', 'the', 'kids'], 'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['£', '20', 'per', 'head'], 'price_range=less than £20')


        apply_rule(i, tokens, tags, 
                   ['a', 'good', 'place', 'to', 'eat'], 
                   'customer_rating=average')

        if tokens[i-3:i+1] == ['the', 'waterman', 'is', 'average']:
            tags[i] = 'customer_rating=average'

        if tokens[i-3:i+1] == ['customers', 'rate', 'NAME', 'highly']:
            tags[i] = 'customer_rating=high'
            tags[i-3] = '0'
            tags[i-2] = '0'

        apply_rule(i, tokens, tags, 
                   ['fair', 'customer', 'ratings'], 
                   'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['1', 'to', '5',], 
                   'customer_rating=1 out of 5')

        apply_rule(i, tokens, tags, 
                   ['the', 'best', 'ratings',], 
                   'customer_rating=high')
        
        apply_rule(i, tokens, tags, 
                   ['lower', '-', 'than', '-', 'average', 'customer', 'rating'], 
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['around', 'twenty', 'pounds', 'per', 'diner'], 
                   'price_range=less than £20')

        apply_rule(i, tokens, tags, 
                   ['perfect', 'on', 'the', 'wallet',],
                   'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['higher', '-', 'end',],
                   'price_range=high')

        if tokens[i-3:i+1] == ['high', 'price', 'average', 'children']:
            tags[i-3] = 'price_range=high'
            tags[i-2] = 'price_range=high'
            tags[i-1] = 'customer_rating=average'

        apply_rule(i, tokens, tags, 
                   ['non', '-', 'family'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['unsuitable', 'for', 'children'],
                   'family_friendly=no')

        if t == 'restaurant':
            tags[i] = 'eat_type=restaurant'



        apply_rule(i, tokens, tags, 
                   ['children', 'are', 'always', 'welcome'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['fairly', 'priced',],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   [ 'price', 'range', 'is', 'moderate', 'rating',], 
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['high', 'priced', 'average', 'meal'], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['high', 'price', 'rate'], 
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['waters', 'edge'], 'area=riverside')

        apply_rule(i, tokens, tags, 
                   ['a', 'children', "'s", 'menu', 'is', 'available', 
                    'suited', 'to', 'most', 'tastes'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['unfriendly', 'family'], 'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['night', 'out', 'without', 'kids'], 'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['not', 'openly', 'welcome', 'families', 'with', 
                    'children'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['on', 'the', 'rivers', 'banks'],
                   'area=riverside')
        apply_rule(i, tokens, tags, 
                   ['bring', 'your', 'children', 'and', 'babies'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['budget', 'food'], 'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['higher', 'than', 'average', 'price'], 'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['extremely', 'modest', 'costs'], 'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'a', 'bit', 'high'], 'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['in', 'the', '£', '20', 'price', 'range'],
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'is', 'slightly', 'above', 'average'],
                   'price_range=high')
        
        apply_rule(i, tokens, tags, 
                   [ 'lower', 'end'],
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['if', 'family', 'dining', 'is', 'what', 'you', 'seek', ',', 'this', 'is', 'not', 'it'], 'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['if', 'family', 'dining', 'is'], '0')
        apply_rule(i, tokens, tags, 
                   ['fair', 'price',], 'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['costs', 'are', 'reasonably', 'low'], 'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'slightly', 'higher', 'than', 'your', 'average'], 'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['uk'], 'food=English')
        apply_rule(i, tokens, tags, 
                   ['quality', 'is', 'amazing'], 'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['an', 'average', ','], 'customer_rating=average')
        
        apply_rule(i, tokens, tags, 
                   ['good', 'customer', 'service'], 'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['north', 'of', 'the', 'city', 'centre'], '0')

        apply_rule(i, tokens, tags, 
                   ["n't", 'rate', 'it', 'highly'], 'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['for', 'average', 'yet'], 'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   [ 'not', 'cater', 'for', 'families'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['kids', 'will', 'love'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['children', 'will', 'be', 'made', 'welcome'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ["n't", 'friendly', 'with', 'children'], 
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['higher', 'cost'], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['highly', 'reviewed'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['north', 'of', 'the', 'city', 'center'],
                   '0')

        apply_rule(i, tokens, tags, 
                   ['not', 'near', 'the', 'city', 'centre'], '0') 

        apply_rule(i, tokens, tags, 
                   ['outside', 'of', 'the', 'city', 'center'], '0') 

        if tokens[i-6:i+1] == ['customers', 'rate', 'the', 'french', 'food', 
                               'here', 'average',]:
            tags[i] = 'customer_rating=average'

        if tokens[i-2:i+1] == ['best', 'english', 'food']:
            tags[i-2] = 'customer_rating=high'

        if tokens[i-3:i+1] == ['a', 'really', 'good', 'choice']:
            for j in range(i-3,i+1):
                tags[j] = 'customer_rating=high'
        apply_rule(i, tokens, tags, 
                   ['non', 'family', 'family'], 
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['not', 'welcoming', 'to', 'children',],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['not', 'a', 'family',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['for', 'adults', 'and', 'kids',],
                   'family_friendly=yes')

        if tokens[i-5:i+1] == ['customer', 'rating', 'for', 'aromi', 'is', 'average']:
            tags[i] = 'customer_rating=average'
        if tokens[i-3:i+1] == ['average', 'rated', 'high', 'price']:
            tags[i-3] = 'customer_rating=average'
            tags[i-2] = 'customer_rating=average'
            tags[i-1] = 'price_range=high'
            tags[i] = 'price_range=high'


        apply_rule(i, tokens, tags, 
                   ['not', 'rated', 'very', 'highly'], 'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['scores', 'highly', 'with', 'customers'], 'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['reviews', 'are', "n't", 'great'], 'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['offers', 'a', 'menu', 'that', 'children', 'would', 'enjoy'], 'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['you', 'need', 'to', 'bring', 'your', 'kids'], 'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['all', 'varieties', 'of', 'age'], 'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['children', 'are', 'welcomed'], 'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['north', 'from', 'the', 'city', 'centre'], '0')

        apply_rule(i, tokens, tags, 
                   ['north', 'of', 'city', 'centre'], '0')
        apply_rule(i, tokens, tags, 
                   [ 'far', 'from', 'the', 'city', 'centre'], '0')
        apply_rule(i, tokens, tags, 
                   ['ratings', 'are', 'below', 'average'],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['customer', 'recommended',],
                   'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['budget', 'prices'],
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['on', 'the', 'edge', 'of', 'the', 'city', 'centre'], '0')

        apply_rule(i, tokens, tags, 
                   ['rated', 'in', 'high', 'value'], 
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['for', 'average', ','], 'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['satisfied', 'client', 'ratings'], 'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['not', 'rated', 'highly'], 'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['scummy', 'food'], 'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['a', '9', 'on', 'a', 'scale', 'of', '1', '-', '10',],
                   'customer_rating=high')
        if tokens[i-2:i+1] == ['fine', 'italian', 'food']:
            tags[i-2] = 'customer_rating=high'

        if tokens[i-3:i+1] == ['moderate', 'priced', 'poor', 'rated']:
            tags[i-3] = 'price_range=moderate'
            tags[i-2] = 'price_range=moderate'
            tags[i-1] = 'customer_rating=low'
            tags[i] = 'customer_rating=low'
        if tokens[i-3:i+1] == ['highly', 'rated', 'average', 'priced']:
            tags[i-1] = 'price_range=moderate'
            tags[i] = 'price_range=moderate'
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'customer_rating=high'


        if tokens[i-6:i+1] == [ 'customer', 'ratings', 'for', 'this', 'restaurant', 'are', 'low']:
            tags[i] = 'customer_rating=low'

        apply_rule(i, tokens, tags, 
                   ['one', '-', 'starred'], 'customer_rating=1 out of 5')

        apply_rule(i, tokens, tags, 
                   ['get', 'top', 'dollar', 'rated'], 'customer_rating=high')

        if t == 'italy':
            tags[i] = 'food=Italian'

        if tokens[i-2:i+1] == ['excellent', 'japanese', 'food']: 
            tags[i-2] = 'customer_rating=high'

        apply_rule(i, tokens, tags, 
                   ["n't", 'have', 'very', 'good', 'reviews'], 'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'of', '3', 'out', 'of'], 'customer_rating=3 out of 5')

        apply_rule(i, tokens, tags, 
                   ['customer', 'ratings', 'for', 'this', 'location', 
                    'are', 'average'], 'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['far', 'from', 'golden', 'in', 'quality'], 
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['low', 'quality', 'food',], 
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['poorly', 'recommended'], 
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['not', 'considered', 'to', 'be', 'a', 'family', '-', 
                    'friendly'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['favored', 'by', 'families',],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['in', 'the', 'core', 'of', 'the', 'city'],
                   'area=city centre')
        apply_rule(i, tokens, tags, 
                   ['not', 'ideally', 'suited', 'for', 'a', 'family', 'meal'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['low', 'in', 'the', 'ratings'], 
                   'customer_rating=low')
        
        apply_rule(i, tokens, tags, 
                   ['a', 'wonderful',], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['no', 'families', 'are', 'allowed'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['can', 'take', 'kids'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['with', 'your', 'kids'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['geared', 'towards', 'adults'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['excellent', 'customer', 'ratings', 'on', 'average'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['food', 'is', 'of', 'the', 'best', 'quality',], 
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ["n't", 'recommended', 'for', 'children',], 
                   'family_friendly=no')
        if tokens[i-3:i+1] == ['highly', 'rated', 'low', 'priced']:
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'customer_rating=high'
            tags[i-1] = 'price_range=low'
            tags[i] = 'price_range=low'

        if tokens[i-4:i+1] == ['highly', 'rated', 'low', '-', 'priced']:
            tags[i-4] = 'customer_rating=high'
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'price_range=cheap'
            tags[i-1] = 'price_range=cheap'
            tags[i] = 'price_range=cheap'

        apply_rule(i, tokens, tags, 
                   ['please', 'keep', 'in', 'mind', 'that', 'the', 
                    'average', 'price', 'range',], 
                   '0')
        apply_rule(i, tokens, tags, 
                   ['less', 'than', 'e', '20'], 'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['for', 'those', 'without', 'children',], 'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['children', 'are', 'not', 'welcomed',], 'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['dishes', 'are', 'slightly', 'pricey'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['reviewers', 'do', "n't", 'recommend'],
                   'customer_rating=low')
        if tokens[i-11:i+1] == ['customers', 'rate', 'this', 'kid', '-', 
                                'friendly', 'pub', ',', 'strada', ',', 'as',
                                'high']:
            tags[i] = 'customer_rating=high'
        if tokens[i-11:i+1] == ['customers', 'rate', 'this', 'kid', '-', 
                                'friendly', 'pub', ',', 'NAME', ',', 'as',
                                'high']:
            tags[i] = 'customer_rating=high'


        if tokens[i-4:i+1] == ['high', 'ratings', 'and', 'great', 'prices']:
            for j in range(i-1,i+1):
                tags[j] = 'price_range=moderate'

        apply_rule(i, tokens, tags, 
                   ['price', 'ranging', 'at', 'a', 'moderate', 'level'], 
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['lower', 'prices', 'than', '£', '20'], 'price_range=less than £20')

        if tokens[i-4:i+1] == ['the', 'golden', 'curry', 'is', 'average']:
            tags[i] = 'customer_rating=average'
        apply_rule(i, tokens, tags, 
                   ['twenty', 'to', 'twenty', '-', 'five', 'dollars'], 
                   'price_range=£20-25')
        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20', '-', '25',], 
                   'price_range=£20-25')

        apply_rule(i, tokens, tags, 
                   ['lower', 'prices', 'to', '£', '20'], 
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['between', 'twenty', 'and', 'twenty', 'five', 'pounds'], 
                   'price_range=£20-25')
        apply_rule(i, tokens, tags, 
                   ['more', 'than', '40'],
                   'price_range=more than £30')

        apply_rule(i, tokens, tags, 
                   ['cost', 'will', 'be', 'high'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['great', 'food', 'and', 'service'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'on', 'the', 'high', 'range'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['rated', 'very', 'favourably', 'by', 'customers'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['highly', 'ranked'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['not', 'favorable', 'for', 'a', 'family',],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['low', 'customer', 'approval', 'rating'],
                   'customer_rating=low')

        if tokens[i-9:i+1] == ['price', 'range', 'at', 'this', 'restaurant',
                               'is', 'slightly', 'higher', 'than', 'average']:
            for j in range(i-4,i+1):
                tags[j] = 'price_range=high'

        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'in', 'the', 'moderate', 'range'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['not', 'interested', 'in', 'family'],
                   'family_friendly=no')
        if tokens[i-3:i+1] == ['expensive', 'but', 'very', 'good',]:
            tags[i-1] = 'customer_rating=high'
            tags[i] = 'customer_rating=high'

        apply_rule(i, tokens, tags, 
                   ['short', 'distance', 'from', 'the', 'city', 'centre',],
                   '0')

        apply_rule(i, tokens, tags, 
                   ['outside', 'of', 'city', 'centre'],
                   '0')

        apply_rule(i, tokens, tags, 
                   [ 'non', 'friendly', 'place'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20', '-', '£', '25'], 
                   'price_range=£20-25')

        apply_rule(i, tokens, tags, 
                   ['not', 'got', 'good', 'ratings'], 
                   'customer_rating=low')
        if tokens[i-5:i+1] == ['customer', 'rating', 'for', 'cocum', 'is', 'average']:
            tags[i] = 'customer_rating=average'

        if tokens[i-17:i+1] == ['customer', 'rating', 'for', 'the', 'french', 
                                'food', ',', 'which', 'cost', 'more', 'than', 
                                '£', '30', ',', 'at', 'strada', 'is', 'low']:
            tags[i] = 'customer_rating=low'
        if tokens[i-17:i+1] == ['customer', 'rating', 'for', 'the', 'french', 
                                'food', ',', 'which', 'cost', 'more', 'than', 
                                '£', '30', ',', 'at', 'NAME', 'is', 'low']:
            tags[i] = 'customer_rating=low'

        apply_rule(i, tokens, tags, 
                   ['high', 'standards', 'average', 'customer', 'reviews'], 
                   'customer_rating=high')
        if tokens[i-10:i+1] == ['it', 'is', 'not', 'the','cheapest', 'the',
                                'quality', 'is', 'not', 'all', 'bad']:
            for j in range(i-10,i-5):
                tags[j] = 'price_range=moderate'
            for j in range(i-5, i+1):
                tags[j] = 'customer_rating=average'

        apply_rule(i, tokens, tags, 
                   ['prices', 'a', 'bit', 'high'],
                   'price_range=high') 
        apply_rule(i, tokens, tags, 
                   [ 'very', 'good', 'reputation'], 
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['mid', 'level', 'reviews'], 
                   'customer_rating=moderate')

        apply_rule(i, tokens, tags, 
                   ['do', 'allow', 'children',], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['caters', 'to', 'families',], 
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['not', 'friend', 'family'], 
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['mid', 'level', 'reviews'], 
                   'customer_rating=moderate')
        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'for', 'you', 'is', 'moderate'], 
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['public', 'house'], 
                   'eat_type=pub')
        apply_rule(i, tokens, tags, 
                   ['prices', 'ranging', 'in', 'the', 'high', 'side',], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['rejects', 'a', 'familial', 'atmosphere'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['rating', 'from', 'customers', 'are', 'average'], 
                   'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['above', 'average', 'price'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['not', 'a', 'place', 'to', 'bring', 'children'], 
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['low', 'rate'], 
                   'customer_rating=low')
        if tokens[i-3:i+1] == ['high', 'rated', 'average', 'priced']:
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'customer_rating=high'
            tags[i-1] = 'price_range=average'
            tags[i] = 'price_range=average'

        apply_rule(i, tokens, tags, 
                   ['high', 'quality', 'food'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['outside', 'the', 'city', 'centre'], 
                   '0')
        apply_rule(i, tokens, tags, 
                   ['high', 'costing'], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['price', 'rang', 'is', 'about', 'average'], 
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['above', 'average', 'priced',], 
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['mid', '-', 'cost'], 'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['low', 'average', 'customer', 'rating'], 
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['great', 'place', 'to', 'eat', 'with', 'children'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['a', 'great', 'place', 'to', 'eat', 'with', 'children'], 
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['a', 'great', 'place', 'for', 'kids'], 
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['ca', "n't", 'bring', 'the', 'whole', 'family'], 
                   'family_friendly=no')
        if tokens[i-5:i+1] == ['price', 'range', 'for', 'wildwood', 'is', 
                               'moderate']:
            tags[i] = 'price_range=moderate'
        apply_rule(i, tokens, tags, 
                   ['adults', 'can', 'enjoy'], 
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['does', 'allow', 'children'], 'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['children', 'most', 'welcome',], 'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['as', 'well', 'as', 'poor', 'customer', 'rating',], 
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['not', 'well', 'reviewed'], 
                   'customer_rating=low')
        if tokens[i-7:i+1] == ['prices', 'are', 'a', 'reflection', 'of', 
                               'the', 'exceptional', 'quality',]:
            for j in range(i-7, i-3):
                tags[j] = 'price_range=high'

        apply_rule(i, tokens, tags, 
                   ['3', 'start', 'reviews'],
                   'customer_rating=3 out of 5')

        apply_rule(i, tokens, tags, 
                   ['catered', 'towards', 'adults'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ["n't", 'friendly', 'to', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['rated', 'quite', 'badly'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['1', 'out', 'of', 'customer', 'rating'],
                   'customer_rating=1 out of 5')
        apply_rule(i, tokens, tags, 
                   ['north', 'of', 'city', '-', 'centre'],
                   '0')
#        apply_rule(i, tokens, tags, 
#                   ['customer', 'satisfaction'], 
#                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['not', 'cater', 'to', 'families'], 'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['children', 'are', 'permitted'], 'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['3', 'out', '5'], 'customer_rating=3 out of 5')
        apply_rule(i, tokens, tags, 
                   ['outskirts', 'of', 'the', 'city', 'centre'], '0')
        apply_rule(i, tokens, tags, 
                   ['rating', 'over', '3', 'to', '5'], 'customer_rating=3 out of 5')
        apply_rule(i, tokens, tags, 
                   ['low', 'costumer', 'rate'], 'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['on', 'a', 'budget'], 'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['great', 'prices'], 'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['above', 'average', 'price'], 'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['prices', 'that', 'are', 'above', 'average'], 'price_range=high')

        if tokens[i-5:i+1] == ['prices', 'and', 'customer', 'ratings', 'are', 
                               'moderate']:
            tags[i-5] = 'price_range=moderate'
            for j in range(i-3,i+1):
                tags[j] = 'customer_rating=average'

    
        apply_rule(i, tokens, tags, 
                   ['customer', 'ratings', 'are', 'moderate'],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'that', 'is', 'high'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'is', 'high'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['average', 'customer', 'service'], 
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['highest', 'ratings', 'by', 'customers'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'in', 'low'], 
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['reasonable', 'ratings'], 
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['outside', 'of', 'the', 'city', 'centre'], 
                   '0')

        apply_rule(i, tokens, tags, 
                   ['ideal', 'place', 'to', 'feed', 'your', 'children',], 
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['not', 'recommended', 'for', 'the', 'whole', 'family'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['no', 'children', 'welcome'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['decently', 'priced',],
                   'price_range=moderate') 

        apply_rule(i, tokens, tags, 
                   ['upper', 'class'], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['less', 'than', 'average', 'cost'], 
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['up', 'that', 'low', 'rating', 'to', 'a', 'high', 'rating',], 
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['fantastic', 'reviews',], 
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['not', 'a', 'family', 'place',], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['20', '£', '-', '30', '£'], 
                   'price_range=£20-25')


        if tokens[i-1:i+1] == ['costly', 'pub']:
            tags[i-1] = 'price_range=high'

        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'very', 'low', 'near', 'to', '20', 'to', '25'], 
                   'price_range=£20-25')

        apply_rule(i, tokens, tags, 
                   ['high', 'in', 'cost',], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   [ 'really', 'low', 'costs'], 
                   'price_range=cheap')
        

        apply_rule(i, tokens, tags, 
                   ['about', 'twenty', 'to', 'twenty', 'five', 'pounds'], 
                   'price_range=£20-25')

        apply_rule(i, tokens, tags, 
                   ["n't", 'break', 'the', 'bank'], 
                   'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['with', 'excellent', 'prices'], 
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['not', 'children', 'family'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['under', 'twenty', 'pounds',], 
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['rating', 'is', 'high'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['well', '-', 'respected'], 
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['not', 'got', 'a', 'good', 'rating'],
                   'customer_rating=low') 
        apply_rule(i, tokens, tags, 
                   ['not', 'have', 'a', 'good', 'customer', 'rating'],
                   'customer_rating=low') 

        apply_rule(i, tokens, tags, 
                   ['not', 'intended', 'for', 'the', 'whole', 'family'], 
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['children', 'welcomed',], 
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['accept', 'all', 'families'], 
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['higher', 'than', 'average', '-', 'priced',], 
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['with', 'high', 'prices'], 
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['highly', '-', 'reviewed'], 
                   'customer_rating=high')

        if tokens[i-3:i+1] == ['moderate', 'price', 'great', 'reviews']:
            tags[i-3] = 'price_range=moderate'
            tags[i-2] = 'price_range=moderate'
            tags[i-1] = 'customer_rating=high'
            tags[i] = 'customer_rating=high'

        if tokens[i-4:i+1] == ['low', 'customer', 'rating', 'average', 
                               'price']:
            tags[i-4] = 'customer_rating=low'
            tags[i-3] = 'customer_rating=low'
            tags[i-2] = 'customer_rating=low'
            tags[i-1] = 'price_range=moderate'
            tags[i] = 'price_range=moderate'

        apply_rule(i, tokens, tags, 
                   ['not', 'have', 'a', 'very', 'good', 'rating'], 
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['almost', 'everyone', 'loves', 'it'], 
                   'customer_rating=average')

        if tokens[i-2:i+1] == ['an', 'average', 'chinese']:
               tags[i-1] = 'customer_rating=average'
        if tokens[i-1:i+1] == ['quality', 'coffee']:
               tags[i-1] = 'customer_rating=high'
        if tokens[i-1:i+1] == ['quality', 'drinks']:
               tags[i-1] = 'customer_rating=high'

        apply_rule(i, tokens, tags, 
                   ['not', 'cater', 'to', 'kids',],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   [ "n't", 'bring', 'your', 'family',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['allow', 'children'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['kids', 'welcome'],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ["n't", 'recommend', 'bringing', 'your', 'family'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['whilst', 'out', 'with', 'the', 'kids',],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['not', 'recommended', 'for', 'families', 'with', 'children'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['poor', 'choice', 'of', 'a', 'place', 'to', 'take', 'a', 'family'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['not', 'oriented', 'toward', 'family',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['no', 'family', 'place',],
                   'family_friendly=no')
        if tokens[i-3:i+1] == ['eagle', 'is', 'average', '.']:
            tags[i-1] = 'customer_rating=average'
        apply_rule(i, tokens, tags, 
                   ['passable'],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['highly', 'rated', 'for', 'its', 'good', 'food'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['not', 'have', 'good', 'reviews'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['a', 'great', 'place'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['lower', 'than', 'average', 'rating',],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['north', 'of', 'cambridge', 'city', 'centre'],
                   '0')

        apply_rule(i, tokens, tags, 
                   ['in', 'the', 'cities', 'centre'],
                   'area=city centre')

        apply_rule(i, tokens, tags, 
                   ['offering', 'good', ','],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['customer', 'ratings', 'are', 'great'],
                   'customer_rating=high')
        if tokens[i-1:i+1] == ['best', 'chinese']:
            tags[i-1] = 'customer_rating=high'
        apply_rule(i, tokens, tags, 
                   ['customers', 'like', 'a', 'lot'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['very', 'well', '-', 'reviewed'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['customers', 'raving', 'about', 'their', 'food'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['customers', 'are', 'raving', 'about', 'their', 'food'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['outskirts', 'of', 'the', 'city', 'center',], 
                   '0')
        apply_rule(i, tokens, tags, 
                   ['a', 'bit', 'pricey'], 
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['average', 'price', 'range', 'is', 'high'], 
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['pricier'], 
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['child', 'friendly', 'and', 'great', 'for', 'adults',
                    'as', 'well'], 
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['moderate', 'prices'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['with', 'high', 'price', 'not', 'low'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['no', 'room', 'for', 'a', 'whole', 'family',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['do', "n't", 'like', 'kids',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['good', 'place', 'to', 'bring', 'the', 'kids',],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ['get', 'away', 'from', 'the', 'busy', 'family'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['not', 'allow', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['prices', 'fall', 'into', 'the', 'high', 'range',],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['for', 'a', 'moderate', 'fee'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'in', 'the', 'average', 'range'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['1', '-', '2', 'stars', 'out', 'of', '5',],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['3', 'point', 'rating', 'out', 'of', '5'],
                   'customer_rating=3 out of 5')

        apply_rule(i, tokens, tags, 
                   ['an', 'average', 'place'],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['an', 'ok', ','],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['3', 'out', '5', 'stars'],
                   'customer_rating=3 out of 5')
        apply_rule(i, tokens, tags, 
                   [ "n't", 'expect', 'great', 'customer', 'service'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['not', 'got', 'the', 'best', 'ratings',],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['customer', 'ratings', 'is', 'so', 'high'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['along', 'the', 'riverbank'],
                   'area=riverside')

        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'are', 'not', 'great'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['little', 'over', '£', '20'],
                   'price_range=£20-25')

        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'in', 'the', 'high', 'range',],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['expensive', 'food', 'in', 'a', 'casual', ',', 'low', 'end', 'atmosphere',],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['4', 'out', 'of', '5', 'stars'], 'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['three', 'start', 'rating'], 'customer_rating=3 out of 5')


        apply_rule(i, tokens, tags, 
                   ['not', 'a', 'good', 'place', 'for', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ["n't", 'allow', 'children'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['child', 'frenziedly'],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['moderate', 'price', 'rating'],
                   'price_range=moderate')
        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'is', 'relative', 'high'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['poor', 'quality', 'food'],
                   'customer_rating=low')

        if tokens[i-4:i+1] == ['average', 'price', 'and', 'rating', 'range']:
            tags[i-4] = 'price_range=moderate'
            tags[i-3] = 'price_range=moderate'
            tags[i-1] = 'customer_rating=average'
            tags[i] = 'customer_rating=average'

        if tokens[i-3:i+1] == ['low', 'priced', 'average', 'italian',]:
            tags[i-3] = 'price_range=cheap'
            tags[i-2] = 'price_range=cheap'
            tags[i-1] = 'customer_rating=average'

        apply_rule(i, tokens, tags, 
                   ['above', 'average', 'in', 'price'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['poor', 'customers', 'feedback',],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20', 'to', '£', '25'],
                   'price_range=£20-25')

        apply_rule(i, tokens, tags, 
                   ['great', 'prices', 'that', 'are', 'very', 'reasonable'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['the', 'best', 'food', 'in', 'the', 'area'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['over', 'thirty', 'pounds'],
                   'price_range=more than £30')

        if tokens[i-2:i+1] == ['excellent', 'french', 'food']:
            tags[i-2] = 'customer_rating=high'

        if tokens[i-2:i+1] == ['good', 'quality', 'coffee']:
            tags[i-2] = 'customer_rating=average'
            tags[i-1] = 'customer_rating=average'
        apply_rule(i, tokens, tags, 
                   [ "n't", 'even', 'allow', 'children',],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['not', 'welcoming', 'to', 'families'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'in', 'the', 'low', 'range'],
                   'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['kids', 'are', 'always', 'friendly',],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['a', 'popular', 'and', 'very', 'well', 'done'],
                   'customer_rating=high')
        if tokens[i-3:i+1] == ['average', 'price', 'average', 'rating']:
            tags[i-3] = 'price_range=moderate'
            tags[i-2] = 'price_range=moderate'
            tags[i-1] = 'customer_rating=average'
            tags[i] = 'customer_rating=average'
        if tokens[i-2:i+1] == ['rate', 'wildwood', 'high']:
            tags[i-2] = '0'
            tags[i] = 'customer_rating=high'
        if tokens[i-2:i+1] == ['rate', 'NAME', 'high']:
            tags[i-2] = '0'
            tags[i] = 'customer_rating=high'
        if tokens[i-7:i+1] == ['customer', 'rating', 'for', 'this', 'coffee', 'shop', 'is', 'low']:
            tags[i] = 'customer_rating=low'
        if tokens[i-2:i+1] == ['an', 'average', 'children']:
            tags[i-1] = 'customer_rating=average'
        if tokens[i-4:i+1] == ['high', 'rated', 'low', '-', 'priced',]:
            tags[i-4] = 'customer_rating=high'
            tags[i-3] = 'customer_rating=high'
            tags[i-2] = 'price_range=low'
            tags[i-1] = 'price_range=low'
            tags[i] = 'price_range=low'
        if t == 'punters':
            tags[i] = 'name=The Punter'

        if tokens[i-3:i+1] == ['low', 'priced', 'high', 'quality',]:
            tags[i-3] = 'price_range=cheap'
            tags[i-2] = 'price_range=cheap'
            tags[i-1] = 'customer_rating=high'
            tags[i] = 'customer_rating=high'

        apply_rule(i, tokens, tags, 
                   ['lower', 'than', 'average', 'ratings'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['good', 'quality', 'drinks'],
                   'customer_rating=average')
        apply_rule(i, tokens, tags, 
                   ['higher', 'class'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['more', 'than', 'l30'],
                   'price_range=more than £30')
        apply_rule(i, tokens, tags, 
                   ['not', 'welcoming', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20', 'and', '£', '25',],
                   'price_range=£20-25')
        apply_rule(i, tokens, tags, 
                   ['not', 'recommended', 'for', 'kids',],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['at', 'a', 'reduced', 'price'],
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'a', 'bit', 'high'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['prices', 'that', 'range', 'high'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20', '-', '50',],
                   'price_range=£20-25')
        apply_rule(i, tokens, tags, 
                   ['by', 'the', 'rive'],
                   'area=riverside')
        if tokens[i-3:i+1] == ['customers', 'rate', 'cotto', 'highly']:
            tags[i-2] = '0'

        if tokens[i-5:i+1] == ['prices', 'for', 'this', 'restaurant', 'are', 'high', ]:
            tags[i] = 'price_range=high'

        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'is', 'a', 'little', 'high'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['price', 'range', 'is', 'a', 'little', 'more', 'than', 'your', 'mom', 'and', 'pop',],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['great', 'quality', 'food'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['rating', 'of', '3',],
                   'customer_rating=3 out of 5')

        apply_rule(i, tokens, tags, 
                   ['serving', 'fantastic'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   [ 'customer', 'rating', 'is', 'so', 'high',],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['customer', 'rate', 'is', 'high'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['offers', 'a', 'mature', 'setting'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['without', 'spending', 'the', 'earth'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'less', 'than', 'expected'],
                   'price_range=cheap')

        apply_rule(i, tokens, tags, 
                   ['to', 'bring', 'the', 'kids'],
                   'family_friendly=yes')

        apply_rule(i, tokens, tags, 
                   ['you', 'can', 'take', 'your', 'kids'],
                   'family_friendly=yes')

        if tokens[i-2:i+1] == ['really', 'good', 'fast',]:
            tags[i-2] = 'customer_rating=high'
            tags[i-1] = 'customer_rating=high'
        apply_rule(i, tokens, tags, 
                   ['customer', 'ratings', 'is', 'high'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['customer', 'rating', 'is', 'very', 'high'],
                   'customer_rating=high')
        apply_rule(i, tokens, tags, 
                   ['excellent', 'service', 'and', 'quality'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['rated', 'in', 'average'],
                   'customer_rating=average')

        apply_rule(i, tokens, tags, 
                   ['serves', 'excellent', 'food'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   [ 'located', 'in', 'city'],
                   'area=city centre')

        apply_rule(i, tokens, tags, 
                   ['must', 'be', 'of', 'legal', 'drinking', 'age'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'very', 'low', 'near', 'to', '20'],
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['price', 'is', 'a', 'little', 'high'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['slightly', 'more', 'than', 'average', 'prices'],
                   'price_range=high')

        apply_rule(i, tokens, tags, 
                   ['up', 'to', '£', '20'],
                   'price_range=less than £20')

        apply_rule(i, tokens, tags, 
                   ['has', 'an', 'average', 'price', 'rating'],
                   'price_range=moderate')

        apply_rule(i, tokens, tags, 
                   ['prices', 'are', 'a', 'little', 'high'],
                   'price_range=high')
        apply_rule(i, tokens, tags, 
                   ['a', '4', 'rating'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['food', 'was', 'excellent'],
                   'customer_rating=high')


        apply_rule(i, tokens, tags, 
                   ['a', 'price', 'range', 'of', '£', '20', 'to', '25',],
                   'price_range=£20-25')
        if tokens[i-6:i+1] == ['not', 'a', 'place', 'to', 'bring', 'the', 'kids']:
            for j in range(i-6,i+1):
                tags[j] = 'family_friendly=no'

        apply_rule(i, tokens, tags, 
                   ['not', 'highly', 'rated'],
                   'customer_rating=low')

        apply_rule(i, tokens, tags, 
                   ['well', '-', 'reviewed'],
                   'customer_rating=high')

        apply_rule(i, tokens, tags, 
                   ['rating', 'of', '1'],
                   'customer_rating=1 out of 5')
        apply_rule(i, tokens, tags, 
                   ['north', 'of', 'the', 'center', 'of', 'the', 'city'],
                   '0')

        apply_rule(i, tokens, tags, 
                   ['range', 'of', '20', 'below'],
                   'price_range=less than £20')

        apply_rule(i, tokens, tags, 
                   ['£', '20', 'or', 'less'],
                   'price_range=less than £20')
        apply_rule(i, tokens, tags, 
                   ['somewhere', 'to', 'eat', 'with', 'children'],
                   'family_friendly=yes')

        if tokens[i-7:i+1] == ['french', 'food', 'is', 'serves', 'in', 'the', 
                               'rice', 'boat']:
            tags[i] = 'name=The Rice Boat'
            tags[i-1] = 'name=The Rice Boat'
            tags[i-2] = 'name=The Rice Boat'
        if tokens[i-9:i+1] == ['for', 'your', 'family', 'at', 'riverside', ',',
                                'in', 'the', 'rice', 'boat',]:
            tags[i] = 'name=The Rice Boat'
            tags[i-1] = 'name=The Rice Boat'
            tags[i-2] = 'name=The Rice Boat'
        if tokens[i-4:i+1] == ['rate', 'the', 'rice', 'boat', 'low']:
            tags[i] = 'customer_rating=low'
        if tokens[i-5:i+1] == ['price', 'range', 'for', 'NAME', 'is', 'moderate']:
            tags[i] = 'price_range=moderate'

        if tokens[i-6:i+1] == ['customer', 'ratings', 'for', 'browns', 'cambridge', 'is', 'average']:
            tags[i] = 'customer_rating=average'


        if tokens[i-3:i+1] == ['for', 'NAME', 'is', 'average']:
            tags[i] = 'customer_rating=average'
        apply_rule(i, tokens, tags, 
                   ['not', 'a', 'good', 'place', 'to', 'take', 'children'],
                   'family_friendly=no')
        apply_rule(i, tokens, tags, 
                   ['can', 'bring', 'your', 'kids'],
                   'family_friendly=yes')
        apply_rule(i, tokens, tags, 
                   ["n't", 'bring', 'your', 'kids'],
                   'family_friendly=no')

        apply_rule(i, tokens, tags, 
                   ['mid', 'range', 'bracket'],
                   'price_range=moderate')
        if tokens[i-3:i+1] == ['the', 'NAME', 'is', 'average']:
            tags[i] = 'customer_rating=average'

        if tokens[i-3:i+1] == ['the', 'eagle', 'is', 'average']:
            tags[i] = 'customer_rating=average'

        if tokens[i-7:i+1] == ['customer', 'rating', 'for', 'the', 'golden', 'palace', 'is', 'average',]:
            tags[i] = 'customer_rating=average'

        apply_rule(i, tokens, tags, 
                   ['low', 'customer', 'approval'],
                   'customer_rating=low')
        if tokens[i-5:i+1] == ['indian', 'cuisine', 'by', 'the', 'rice', 'boat']:
            tags[i] = 'name=The Rice Boat'
            tags[i-1] = 'name=The Rice Boat'

        apply_rule(i, tokens, tags, 
                   ['1', 'our', 'of', '5'],
                   'customer_rating=1 out of 5')
        if tokens[i-3:i+1] == ['the', 'wrestlers', 'is', 'average']:
            tags[i] = 'customer_rating=average'

        apply_rule(i, tokens, tags, 
                   ['low', 'costs'],
                   'price_range=cheap')
        apply_rule(i, tokens, tags, 
                   ['has', 'a', 'children', 'area'],
                   'family_friendly=yes')
        if tokens[i-9:i+1] == ['available', 'in', 'the', 'rice', 'boat', 'near', 'express', 'by', 'holiday', 'inn']:
            tags[i-3] = 'near=Express by Holiday Inn'
            tags[i-2] = 'near=Express by Holiday Inn'
            tags[i-1] = 'near=Express by Holiday Inn'
            tags[i] = 'near=Express by Holiday Inn'
            tags[i-5] = 'name=The Rice Boat'
            tags[i-6] = 'name=The Rice Boat'
            tags[i-7] = 'name=The Rice Boat'
             

        apply_rule(i, tokens, tags, 
                   ['average', 'low', 'customer', 'rating'],
                   'customer_rating=low')
        if tokens[i-2:i+1] == ['pricey', '3', 'star']:
            tags[i-2] = 'price_range=high'
        if tokens[i-5:i+1] == ['the', 'prices', 'at', 'fitzbillies', 'are', 'low']:
            for j in range(i-5,i+1):
                tags[j] = '0'
            tags[i] = 'price_range=cheap'
            tags[i-2] = 'name=Fitzbillies'
        if tokens[i-5:i+1] == ['the', 'prices', 'at', 'NAME', 'are', 'low']:
            for j in range(i-5,i+1):
                tags[j] = '0'
            tags[i] = 'price_range=cheap'
            tags[i-2] = 'name=PLACEHOLDER'

        apply_rule(i, tokens, tags, 
                   [ "n't", 'have', 'good', 'customer', 'ratings'],
                   'customer_rating=low')
        apply_rule(i, tokens, tags, 
                   ['are', 'family', '-', 'friendly'],
                   'family_friendly=yes')



        if tokens[i-2:i+1] == ['low', 'quality', 'coffee']:
            tags[i-2] = 'customer_rating=low'
            tags[i-1] = 'customer_rating=low'
    if tokens == ['café']:
        tags[0] = 'eat_type=coffee shop'

    numrats = [x.endswith('out of 5') for x in tags]
    if True in numrats:
        nridx = numrats.index(True)
        in_prev = False
        for j in range(nridx -1, -1, -1):
            if not in_prev and tags[j] == '0':
                continue
            elif not in_prev and tags[j] != '0':
                in_prev = True
            
            if tags[j].startswith('customer_rating'):
                tags[j] = '0'
            else:
                break

        rnridx = len(numrats) - 1 - numrats[::-1].index(True)
        in_next = False
        for j in range(rnridx + 1, len(numrats)):
            if not in_next and tags[j] == '0':
                continue
            elif not in_next and tags[j] != '0':
                in_next = True

            if tags[j].startswith('customer_rating'):
                tags[j] = '0'
            else:
                break

                    
    numrats = ['£' in x for x in tags]
    if True in numrats:
        nridx = numrats.index(True)
        in_prev = False
        for j in range(nridx -1, -1, -1):
            if not in_prev and tags[j] == '0':
                continue
            elif not in_prev and tags[j] != '0':
                in_prev = True
            
            if tags[j].startswith('price_range'):
                tags[j] = '0'
            else:
                break
        rnridx = len(numrats) - 1 - numrats[::-1].index(True)
        in_next = False
        for j in range(rnridx + 1, len(numrats)):
            if not in_next and tags[j] == '0':
                continue
            elif not in_next and tags[j] != '0':
                in_next = True

            if tags[j].startswith('price_range'):
                tags[j] = '0'
            else:
                break



    return tags 

def apply_rule(i, tokens, tags, cond, label): 
    l = len(cond)
    if tokens[i-l+1:i+1] == cond:
        for j in range(i-l+1,i+1):
            tags[j] = label
