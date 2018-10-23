
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import pickle


# In[55]:


data = pickle.load(open('DATA.dat','rb'))


# In[57]:


len(data)


# In[58]:


ref_ans = ['You need to know how much vinegar was used in each container. You need to know what type of vinegar was used in each container. You need to know what materials to test. You need to know what size/surface area of materials should be used. You need to know how long each sample was rinsed in distilled water.You need to know what drying method to use. You need to know what size/type of container to use. Other acceptable responses.' , 'Plastic sample B has more stretchability than the other polymer plastics. Plastic sample A has the least amount of stretchability compared to the other polymer plastics. Not all polymer plastics have the same stretchability. Different polymer plastics have different stretchability and are therefore suited for different applications. A reasonable conclusion cannot be drawn due to procedural errors. Provide the before and after measurements for length. Did the samples all start out the same size?. Make sure the samples are all of the same thickness. Variations in thickness could have caused variations in stretchability. Perform additional trials. Some of the samples have similar stretchability A and C, B and D. Two trials may not be enough to conclusively state that one is more stretchable than the other. Indicate how many weights were added to the clamps. Was it the same number for each sample?.','''BUSHNELL, Fla—RobRoy Maclnnes is the man to see if you want to buy a crocodile. Or a scorpion, a rattlesnake, a boa constrictor. Got a hankering for a cobra? Just pony up $600 and you can have one of the more lethal species."It is a very effective threat display," Maclnnes, 49, says as a Pakistan black cobra, six feet long, hissing, hood spread, writhes in its enclosure and strikes again and again and again at the thin glass separating the creature from a visitor. "A snake like that, coining at you, you would leave him alone." Or simply die of fright. Maclnnes is co-owner of Glades Herp Farms, an empire of claws, spines, scales, fangs and darting tongues. The reptile trade, he is happy to report, is booming. The pet industry estimates that about 4.8 million households now contain at least one pet reptile, a number that has nearly doubled in a decade. Reptiles are increasingly popular in a crowded, urbanized nation. They do not need a yard. You do not have to take a lizard for a walk. But biologists see the trade in nonnative creatures as a factor in the rising number of invasive species, such as the Burmese python, which is breeding up a storm in the Everglades, and the Nile monitor lizard, a toothy carnivore that can reach seven feet in length and has found a happy home along the canals of Cape Coral. Under a new state law, a customer must obtain a $100 annual permit to buy a monitor lizard or some of the largest snakes-four species of pythons and the green anaconda. The animal must also be implanted with a microchip. That tag could help officials identify the animal if it turns up later in the wild. Maclnnes contends that the government overestimates the threat posed by invasive reptiles. He says he is being blocked by the U.S. Fish and Wildlife Service from importing some commercially attractive animals, such as Fiji island iguanas and radiated tortoises from Madagascar. Even the term "invasive species" is unfair, he said. "They are introduced." I think that invasive is passing judgment.Of the pythons, he said: "To me, it is a wonderful introduction. I think it is the best thing to happen to the Everglades in the last 200 years." Biologists, however, say that invasive species, unchecked by natural predators, are major threats to biodiversity. Life on Earth has always moved around, but never so fast. Organisms evolve in niche environments. What happens when the natural barriers are removed? When anything can go anywhere?Complications ensue. Skip Snow, a wildlife biologist for Everglades National Park, has helped drag hundreds of Burmese pythons out of the weeds, of roadways and even from under the hood of a tourist's car. He calls Maclnnes's argument "ridiculous." The snakes, he says, are imperiling five endangered species in the Florida Keys, including the Key Largo wood rat, one specimen of which, tagged with a radio transmitter, was tracked all the way to the belly of a python. No one knows how the snakes went native, but there's speculation that Hurricane Andrew, which obliterated thousands of homes, played a factor in a wholesale python jailbreak in 1992. Many invasive species undergo a lag before proliferating. What's certain is that, by 2002, pythons were seen in multiple locations in remote regions of the Everglades. Then one morning in early 2003 a bunch of tourists on the park's Anhinga Trail, a reliable location for viewing wildlife, were startled to see an alligator with a python in its mouth. Even more dramatic was what happened in the Everglades in 2005: A python swallowed an alligator and—there’s not a delicate way to put it—exploded. The photograph ran around the world; it wasn't pretty, but you had to look. This February, the U.S. Geological Survey reported that pythons in Asia inhabit climates that are similar to those in about a third of the continental United States. A USGS map showed potential python habitat stretching from California to Delaware and including much of the South. You could conceivably have pythons snacking their way right up the Potomac. The map wasn't a prediction of where the snakes will actually spread, however. Media coverage of it was overly sensational, argues the map’s co-author, Robert Reed. "When was the last snake story that didn't get sensationalized?" he asked. "Ecophobia is playing a role," said Jamie K. Reaser, a science and policy adviser to the Pet Industry Joint Advisory Council. "Mammals are warm and fuzzy. Birds tend to have quite a following. But animals such as lizards and snakes tend, at least in this culture, to be less well respected or supported." What is happening in Florida illustrates a broader fact about life on Earth: We live in an age that favors generalists rather than specialists. A generalist is a raccoon, a python, a cockroach, a white-tailed deer. The ultimate generalist is, arguably, a human being, who with the assistance of technology can live anywhere from Florida to Antarctica to outer space. It's no accident that the species that have become most abundant are often those that do best in and around humans. A specialist is China's panda, which eats almost nothing but bamboo, or Australia's koala bear, which eats eucalyptus leaves almost exclusively. Maclnnes is not without an environmental conscience. "We're degrading the Earth at an alarming rate," he said. "Will man go extinct before we reach the point where we figure it out?" He added: "What favors generalists is change. What favors specialists is stability. Right now, mankind has chosen to make Earth a rapidly changing place." Down in the Everglades, Skip Snow would agree with that part of Maclnnes's philosophy. We are all part of a vast experiment in the blending of organisms from around the world, he said. "The thing about the experiment is, it's not planned, and there's no one in control," Snow added. "It's an experiment run amok."''', '''BUSHNELL, Fla—RobRoy Maclnnes is the man to see if you want to buy a crocodile. Or a scorpion, a rattlesnake, a boa constrictor. Got a hankering for a cobra? Just pony up $600 and you can have one of the more lethal species."It is a very effective threat display," Maclnnes, 49, says as a Pakistan black cobra, six feet long, hissing, hood spread, writhes in its enclosure and strikes again and again and again at the thin glass separating the creature from a visitor. "A snake like that, coining at you, you would leave him alone." Or simply die of fright. Maclnnes is co-owner of Glades Herp Farms, an empire of claws, spines, scales, fangs and darting tongues. The reptile trade, he is happy to report, is booming. The pet industry estimates that about 4.8 million households now contain at least one pet reptile, a number that has nearly doubled in a decade. Reptiles are increasingly popular in a crowded, urbanized nation. They do not need a yard. You do not have to take a lizard for a walk. But biologists see the trade in nonnative creatures as a factor in the rising number of invasive species, such as the Burmese python, which is breeding up a storm in the Everglades, and the Nile monitor lizard, a toothy carnivore that can reach seven feet in length and has found a happy home along the canals of Cape Coral. Under a new state law, a customer must obtain a $100 annual permit to buy a monitor lizard or some of the largest snakes-four species of pythons and the green anaconda. The animal must also be implanted with a microchip. That tag could help officials identify the animal if it turns up later in the wild. Maclnnes contends that the government overestimates the threat posed by invasive reptiles. He says he is being blocked by the U.S. Fish and Wildlife Service from importing some commercially attractive animals, such as Fiji island iguanas and radiated tortoises from Madagascar. Even the term "invasive species" is unfair, he said. "They are introduced." I think that invasive is passing judgment.Of the pythons, he said: "To me, it is a wonderful introduction. I think it is the best thing to happen to the Everglades in the last 200 years." Biologists, however, say that invasive species, unchecked by natural predators, are major threats to biodiversity. Life on Earth has always moved around, but never so fast. Organisms evolve in niche environments. What happens when the natural barriers are removed? When anything can go anywhere?Complications ensue. Skip Snow, a wildlife biologist for Everglades National Park, has helped drag hundreds of Burmese pythons out of the weeds, of roadways and even from under the hood of a tourist's car. He calls Maclnnes's argument "ridiculous." The snakes, he says, are imperiling five endangered species in the Florida Keys, including the Key Largo wood rat, one specimen of which, tagged with a radio transmitter, was tracked all the way to the belly of a python. No one knows how the snakes went native, but there's speculation that Hurricane Andrew, which obliterated thousands of homes, played a factor in a wholesale python jailbreak in 1992. Many invasive species undergo a lag before proliferating. What's certain is that, by 2002, pythons were seen in multiple locations in remote regions of the Everglades. Then one morning in early 2003 a bunch of tourists on the park's Anhinga Trail, a reliable location for viewing wildlife, were startled to see an alligator with a python in its mouth. Even more dramatic was what happened in the Everglades in 2005: A python swallowed an alligator and—there’s not a delicate way to put it—exploded. The photograph ran around the world; it wasn't pretty, but you had to look. This February, the U.S. Geological Survey reported that pythons in Asia inhabit climates that are similar to those in about a third of the continental United States. A USGS map showed potential python habitat stretching from California to Delaware and including much of the South. You could conceivably have pythons snacking their way right up the Potomac. The map wasn't a prediction of where the snakes will actually spread, however. Media coverage of it was overly sensational, argues the map’s co-author, Robert Reed. "When was the last snake story that didn't get sensationalized?" he asked. "Ecophobia is playing a role," said Jamie K. Reaser, a science and policy adviser to the Pet Industry Joint Advisory Council. "Mammals are warm and fuzzy. Birds tend to have quite a following. But animals such as lizards and snakes tend, at least in this culture, to be less well respected or supported." What is happening in Florida illustrates a broader fact about life on Earth: We live in an age that favors generalists rather than specialists. A generalist is a raccoon, a python, a cockroach, a white-tailed deer. The ultimate generalist is, arguably, a human being, who with the assistance of technology can live anywhere from Florida to Antarctica to outer space. It's no accident that the species that have become most abundant are often those that do best in and around humans. A specialist is China's panda, which eats almost nothing but bamboo, or Australia's koala bear, which eats eucalyptus leaves almost exclusively. Maclnnes is not without an environmental conscience. "We're degrading the Earth at an alarming rate," he said. "Will man go extinct before we reach the point where we figure it out?" He added: "What favors generalists is change. What favors specialists is stability. Right now, mankind has chosen to make Earth a rapidly changing place." Down in the Everglades, Skip Snow would agree with that part of Maclnnes's philosophy. We are all part of a vast experiment in the blending of organisms from around the world, he said. "The thing about the experiment is, it's not planned, and there's no one in control," Snow added. "It's an experiment run amok."''', '''mRNA exits nucleus via nuclear pore.  mRNA travels through the cytoplasm to the ribosome or enters the rough endoplasmic reticulum. mRNA bases are read in triplets called codons (by rRNA). tRNA carrying the complementary (U=A, C+G) anticodon recognizes the complementary codon of the mRNA. The corresponding amino acids on the other end of the tRNA are bonded to adjacent tRNA’s amino acids. A new corresponding amino acid is added to the tRNA. Amino acids are linked together to make a protein beginning with a START codon in the P site (initiation). Amino acids continue to be linked until a STOP codon is read on the mRNA in the A site (elongation and termination).''' ,'''Selective permeability is used by the cell membrane to allow certain substances to move across. Passive transport occurs when substances move from an area of higher concentration to an area of lower concentration. Osmosis is the diffusion of water across the cell membrane. Facilitated diffusion occurs when the membrane controls the pathway for a particle to enter or leave a cell. Active transport occurs when a cell uses energy to move a substance across the cell membrane, and/or a substance moves from an area of low to high concentration, or against the concentration gradient. Pumps are used to move charged particles like sodium and potassium ions through membranes using energy and carrier proteins. Membrane-assisted transport occurs when the membrane of the vesicle fuses with the cell membrane forcing large molecules out of the cell as in exocytosis. Membrane-assisted transport occurs when molecules are engulfed by the cell membrane as in endocytosis. Membrane-assisted transport occurs when vesicles are formed around large molecules as in phagocytosis. Membrane-assisted transport occurs when vesicles are formed around liquid droplets as in pinocytosis. Protein channels or channel proteins allow for the movement of specific molecules or substances into or out of the cell.''', '''Rose’s head jerked up from her chest. “Oh no,” she groaned, rubbing the back of her neck and blinking at the bright light in the kitchen. For a split second she was confused. Then she remembered: her essay for the state competition. She’d been struggling to think of a topic. Her brain must have surrendered to exhaustion.The day, like most of her days, had been too long, too demanding. From school she’d gone straight to the restaurant to work a four-hour shift, then straight home to help Aunt Kolab prepare a quick supper. After that it was time to do homework.  When would she squeeze in writing a flawless three-thousand-word essay? “I’m insane,” she said grimly as she gathered books and papers.  Even if I win, she thought, I won’t get to travel to Sacramento to receive the prize. She’d already had to miss a lot of shifts, and her supervisor was on the verge of firing her. Her younger sister walked in rubbing her eyes.  “Anna,” Rose said. “What’s wrong? You feel okay?” “I’m fine,” her sister said.  “I just had another bad dream.” “I fell asleep working on my essay,” Rose said.  Anna poured two glasses of orange juice and handed one to Rose. “Mama’s not home yet, is she.” It wasn’t a question. “I hate how late she has to work.” Her voice sank to a fierce whisper. “I’m so lonesome for Papa. It seems like he’s been gone for years.” “It’s only been four months,” Rose said as gently as she could. “He had to go. The job in Los Angeles paid three times what he was making here.” Anna glared at Rose.  “Money isn’t everything.”  “Only if you already have everything,” Rose said. She tried a laugh that sounded fake even to her. “We have our part to do to help Paul finish  college. Then he’ll get a good job, Anna, and he’ll pay for you and me to go to college.” Anna rolled her eyes and shoved her chai r away from the table. “You sound just like Mama.” She stood and stalked out of the kitchen. By the time Rose tiptoed into their room, Anna was already snoring lightly. Rose slid into bed and watched the lights from passing cars move across the walls. They became the lights that had illuminated the stage at Paul’s high school graduation. As her brother accepted his diploma, Rose had glanced at her parents’ faces. Four eyes shining with tears. The work, the sheer weight of it, to get him on that stage slid from them in that moment; only a sweet, triumphant ache remained. Surely they remembered the ship, their young son and daughters clinging to their necks, Cambodia behind them, the United States before them. On that ship perhaps they had imagined their children’s futures, imagined this very day would come. In the dark Rose clasped then cupped her hands.  Paul’s fate lies partly in these, she thought. She felt too young for so much responsibility. Then she shivered, imagining how her brother must feel. Only three years older, he held the fate of two people—both his sisters—in his hands. Rose dreamed that she swam through clear, green-tinted water, enjoying the pure simplicity of a fish’s life. She stopped moving and looked up. She saw Paul jump from a boulder and crash into the water just above her. His body sank as if it were made of stone, pushing her beneath him down to the sandy bottom. She struggled to get out from under him, but he seemed unaware of her. When she opened her mouth to scream get off, water rushed in. Rose woke gasping for air. The walls of her room were bathed in pale sunlight. When her heart had slowed back down, she got up. Anna was still asleep. In the hall Rose stopped at her mother’s room. She was also sleeping. So it was Aunt Kolab making the muted noises coming from the kitchen. “Good morning, Rose,” her aunt said. Rose felt an urgent need to relate the dream, to expose it so it would loosen its grip on her. After she’d finished, her aunt said, with a puzzled look, “Do you feel so weighed down by what you’re doing to help this family?” Rose didn’t answer. If she told the truth, she would hurt her aunt. And probably her aunt would tell her mother.  “In Cambodia, our first country, what we’re all doing would be quite normal,” her aunt said. “But now I realize that you’re seeing the situation through other eyes—as you should, I suppose, because you grew up here. This must be difficult for you. Yes?” Rose nodded. “Hmm. Maybe we can find a way to do things differently. A way better for you.” Her aunt’s face lit up. “Maybe I can sew for ladies. Or I could make special treats from our country and sell them.” Rose kept nodding. Maybe her life would get easier. Maybe it wouldn’t. But her aunt’s offer had somehow made her feel lighter. Suddenly, it occurred to her: here was the topic for her essay, although it was still vague. Cambodian tradition and sense of family, she realized, could survive an ocean crossing.''','''I met Mr. Leonard when I started middle school. He was a hall monitor whose job it was to keep students moving along from one classroom to the next. “Move along, people, move along!” he’d advise the shuffling crowd, and everyone complied. I distinguished myself from the masses by being one of a select few in the remedial reading program. Twice a week, I left English class early for the learning center in the basement, where I worked with a tutor. On my first trip, Mr. Leonard confronted me in the stairwell. “Hey, my friend, where do you think you’re going?” he asked, arms folded across his chest.“Learning center,” I muttered, showing him my hall pass.“Why?” he asked from behind a hard stare.“Why?” I answered automatically, “I can’t read.” His gaze softened. “Fair enough. On your way, then. Work hard.” For the next few weeks, that was the extent of our conversations. He’d meet me in the stairwell, I’d show him my pass. Then one day he surprised me by asking what I did after school. “Nothing,” I answered. “Just some homework.”“Meet me in the gym. 2:30.” Since this gave me a legitimate reason to delay my daily homework battles, I agreed. When I arrived, the gym was crowded with kids warming up for intramurals. Mr. Leonard was seated in a corner, watching. He waved me over, then pointed at the kids chasing basketballs. “None of this appeals to you?” he asked. I shook my head. When you’re the last guy chosen for teams in gym class, you don’t seek out more of that treatment after school. “Follow me,” he directed, and, obediently, I followed. We left the building and went to the track. Spread along the inside lane were hurdles. Mr. Leonard pointed at the closest one.“Know what that thing is called?” he asked. “A hurdle,” I answered.“Know what to do with it?” he questioned. “You jump it,” I replied.“Well?” he responded. “On your way then.” It never occurred to me to refuse--perhaps I’d been conditioned by hearing those words every day. I got into a slow jog and awkwardly hopped over each barrier for a whole lap.“Not a bad first effort,” commented Mr. Leonard as I staggered in. “That was terrible,” I gasped.“You’ll do better next time,” he responded. “Bring sneakers and shorts tomorrow.”“Right,” I panted. Mr. Leonard began walking back toward the school, then turned and asked, “Say, what’s your name?”“Paul.” It didn’t occur to me until later that this was an odd question for someone who had checked my hall pass twice a week. And so it began. Monday through Friday, rain or shine, I was out on the track with Mr. Leonard shouting from the side. “Open your stride!” “Pump your arms!” “Lean, . . .  NOW!”  I improved steadily until one day I found myself standing before the high school track coach. “How’d you get so fast, son?” he asked. “Well, I’ve been training,” I replied. “Someone’s helping me.”“Mr. Leonard Grabowski?” I nodded. The coach smiled and asked me to work out with the high school team. Then he scribbled on a scrap of paper and handed it me. It was a URL for a track and field website. “Visit this site. Do a search for ‘Grabowski.’” The next day, I told Mr. Leonard about my conversation with the coach and asked if he thought I should work out with the team. “Absolutely,” he replied with a grin. “A little competition will only help.” I pulled the printout I’d downloaded the night before from my pocket. “Why didn’t you tell me about this?” He looked at me quizzically, then smiled sadly at the image on the page. “I looked good back then, didn’t I?” he chuckled. I moved beside him and pointed to the photograph. “You were a college freshman who won the 400 meter hurdles at the nationals. You broke records.” “I remember,” he said solemnly. “Best race of my life.” “Well, what happened after that?” I pressed.Mr. Leonard handed the paper back and looked at the ground, his brow furrowed, his voice cracked as he spoke.“I was a good athlete,” he said softly, “but not a good student.  We had no learning centers in our school. I relied on friends to help me get by, but even then the work was always too hard.”  His voice trailed off. “But you went to college,” I said.“Things were different back then,” he replied. “The college scouts told me that my grades didn’t matter, that I’d have tutors to help me, but college work is a whole lot harder than high school work. I lost my scholarship and flunked out. No other school wanted a runner who couldn’t read.” The emotions in Mr. Leonard’s words were all too familiar to me. I knew them well--feelings of embarrassment when I was called upon to read aloud or when I didn’t know an answer everyone else knew. This man had given his time to help me excel at something. Suddenly I realized what I could do for him. “C’mon, Mr. Leonard,” I said, walking back toward school. “It’s time to start your training.”''','''Grab your telescope! Look up in the sky! It’s a comet! It’s a meteor! It’s a tool bag?Such an observation isn’t as strange as it seems. Orbital pathways around our planet that were once clear are now cluttered with the remains of numerous space exploration and satellite missions. This “space junk” is currently of great concern to government space agencies around the globe.What Is Space Junk?In 1957, the Soviet Union launched the first artificial satellite. The United States followed suit, and thus began the human race’s great space invasion.Over the past 52 years, a variety of spacecraft, including space capsules, telescopes, and satellites, have been sent beyond Earth’s atmosphere. They explore the vast reaches of our solar system, monitor atmospheric conditions, and make global wireless communication possible. The rockets that are used to power these spacecraft typically fall back to Earth and disintegrate in the intense heat that results from friction with Earth’s atmosphere. The objects themselves, however, are positioned hundreds of miles above Earth, far from elements that would cause them to degrade or burn up. In this airless environment, some of them continue to circle the planet indefinitely. While this is ideal for a fully functioning object that was launched for that purpose—for example, a communications satellite—what happens when a satellite “dies” or malfunctions and can’t be repaired? The disabled object becomes a piece of high-tech junk, circling the globe in uncontrolled orbit.With no one at the controls, dead satellites run the risk of colliding with each other. That’s exactly what happened in February 2009. Two communications satellites, one American and one Russian, both traveling at more than 20,000 miles per hour, crashed into each other 491 miles above the Earth. The impact created hundreds of pieces of debris, each assuming its own orbital path. Now, instead of two disabled satellites, there are hundreds of microsatellites flying through space. It’s not only spectacular crashes that create debris. Any objects released into space become free-orbiting satellites, which means that astronauts must take great care when they leave their spacecraft to make repairs or do experiments. Still, accidents do happen: in 2008, a tool bag escaped from the grip of an astronaut doing repairs on the International Space Station (ISS). Little Bits, But a Big Deal. So who cares about a lost tool bag or tiny bits of space trash? Actually, many people do. Those bits of space debris present a very serious problem. Tiny fragments traveling at a speed of five miles per second can inflict serious damage on the most carefully designed spacecraft. If you find that hard to believe, compare grains of sand blown by a gentle breeze to those shot from a sandblaster to strip paint from a concrete wall. At extreme speeds, little bits can pack a punch powerful enough to create disastrous holes in an object moving through space. Scientists are hard-pressed for an easy solution to the problem of space junk. Both the National Aeronautics and Space Agency (NASA) and the European Space Agency maintain catalogues of known objects. The lost tool bag, for example, is listed as Satellite 33442. But while military radar can identify objects the size of a baseball, anything smaller goes undetected. This makes it difficult for spacecraft to steer clear of microdebris fields. Accepting the inevitability of contact, engineers have added multiple walls to spacecraft and stronger materials to spacesuits to diminish the effects of impact.Yet the problem is certain to persist. In fact, the amount of space trash is actually increasing because commercial space travel is on the rise and more nations have undertaken space exploration. Space agencies hope that the corporations and nations involved can work together to come up with a viable solution to space pollution. ''','''Black. The doghouse will be warmer. The black lid made the jar warmest. Dark gray. The inside will be a little warmer, but not too hot. The dark gray lid increased 6º C more than the white. Light gray. The inside will stay cooler, but not too cool. The light gray lid was 8º C cooler than the black. White. The inside will be cooler. The white lid only went up to 42º C.''']


# In[59]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# encode a list of lines
def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# In[60]:


max_length = 1808
max_length_1 = 5875


# In[61]:


vocab_size = 16172


# In[62]:


embedding_matrix_glove = pickle.load(open('GLOVE_MATRIX.dat','rb'))


# In[63]:


from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
callbacks = [
     EarlyStopping(monitor='val_acc' , patience = 5 , verbose=1)
    #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import ml_metrics as m


# In[67]:



def dot_product(x, kernel):
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.

        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
#        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


# In[68]:


import re
import os as os
import numpy as np
import itertools
from sklearn import preprocessing


# In[69]:


def convert_labels (trainY):
    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    temp1 = le.transform(trainY)
    return to_categorical(temp1,4), le.classes_, trainY

_,lable_encoding,_=convert_labels(data['y'])

def get_class_from_pred(pred):
    return [lable_encoding[x.argmax()] for x in pred]


# In[70]:


def define_model(learning_rate,dropout,lstm_out,n_hidden_layer,em,em_trainable_flag):
    input_sentence= Input(shape=(max_length,),name='Sentence')    
    input_reference = Input(shape=(max_length_1,),name='Reference')
    
    embedding=Embedding(vocab_size, len(eval(em)[0]), weights = [eval(em)],input_length=max_length,trainable = False)
    emb_ref=Embedding(vocab_size, len(eval(em)[0]), weights = [eval(em)],input_length=max_length_1,trainable = False)

    context = embedding(input_sentence)
    reference = emb_ref(input_reference)
    
    combined= concatenate([context, reference], axis = 1)
    combined=Dropout(0.5)(combined)
    c = Conv1D(150,5,activation='relu')(combined)
    
    hidden,_,_,_,_ = Bidirectional(LSTM(300, return_sequences=True, return_state = True, dropout=0.25, recurrent_dropout=0.1))(c)
    
    a = Attention()(hidden)
    #a = AveragePooling1D(a)

    x=Dense(300,activation='relu')(a)
        
    output=Dense(4,activation='softmax')(x)

    model= Model(inputs=[input_sentence,input_reference] ,outputs=output)
    
    optimizer = Adam(lr=learning_rate)
    
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    

    
    return model

from keras.layers import Lambda,Reshape,concatenate,Input, Embedding, LSTM


# In[ ]:


from sklearn.model_selection import train_test_split
count=0;
test_count=0;
cvscores = []
final_Core= [] 
ID=[]
NUMBER=[]

for i in np.unique(data['ID']):

        #print (train_index , test_index)
    X_train, X_test, y_train, y_test = train_test_split(data['X_token'],data['y'],test_size = 0.2 , random_state=0)
    X_train, X_test, X_ref_train, X_ref_test = train_test_split(data['X_token'],data['X_ref'],test_size = 0.2 , random_state=0)


  #  print (X_train.shape, y_train.shape)
    y_train = to_categorical(y_train , 4)
    y_test = to_categorical(y_test , 4)

    model = define_model(learning_rate=0.001,
                     dropout=0.5,
                     lstm_out=300,
                     n_hidden_layer=1,
                     em='embedding_matrix_glove',
                     em_trainable_flag=False
                    )

    EarlyStop= EarlyStopping(monitor='val_loss',patience=5,verbose=1)

    
    model.fit(x=[X_train,X_ref_train],y=y_train, epochs=100,batch_size=32,
              callbacks=[EarlyStop],validation_data=([X_test,X_ref_test],y_test))

    pred = model.predict([X_test,X_ref_test])
    out= get_class_from_pred(pred)
    actual= get_class_from_pred(y_test)

    print('Kappa: ', m.quadratic_weighted_kappa(actual,out))

