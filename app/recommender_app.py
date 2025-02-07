import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
from utils.functions import *


# Load models and scaler only once

spotify_scaler = joblib.load('../scaler/spotify_scaler.pkl')
kmeans_model = joblib.load('../models/kmeans_model.pkl')

# Set your Spotify credentials
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')


# Initialize Spotify client
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))



# Initialize session state for page management and data persistence
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None
if 'tracks_info' not in st.session_state:
    st.session_state.tracks_info = None
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'rejected_songs' not in st.session_state:
    st.session_state.rejected_songs = set()
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = None

# Page navigation functions
def on_find_songs_click():
    st.session_state.page = 'validation'

def on_use_this_click(song):
    # Store selected song details in session state
    st.session_state.selected_song = song
    st.session_state.page = 'recommendations'

def on_new_song_click():
    st.session_state.page = 'landing'

def reject_song(song_name):
    # Add the rejected song to our set
    st.session_state.rejected_songs.add(song_name)
    # Get new recommendation (you'll need to implement get_new_recommendation())
    new_rec = get_new_recommendation(st.session_state.selected_song, st.session_state.rejected_songs)
    
    # Replace the rejected song with the new one
    for i, rec in enumerate(st.session_state.current_recommendations):
        if rec['name'] == song_name:
            st.session_state.current_recommendations[i] = new_rec
            break



#  Start with the first page (Landing page)
if st.session_state.page == 'landing':
    st.title('Spotify Song Recommender')
    st.write('Find your next favorite song based on your music taste!')
    
    # User inputs
    query = st.text_input('Enter a song or artist name')
    
    # Popularity filter (0-100)
    popularity = st.slider('Select popularity', 0, 100, (0, 100), step=1)
    
    # Year filter (1900 - 2024)
    year = st.slider('Select year range', 1900, 2024, (1900, 2024), step=1)

    # Checkbox for hot songs only
    hot_songs_only = st.checkbox('Use hot songs only!')

    def search_and_go():
        if query:
            # Call search_song with the input query and filters applied
            tracks_info = search_song(query, (popularity[0], popularity[1]), (year[0], year[1]), hot_songs_only)
            
            if tracks_info is not None:
                st.session_state.tracks_info = tracks_info  # Store the search results in session state
                st.session_state.page = 'validation'  # Move to validation page
            else:
                st.write("No songs found. Try searching with different terms.")

    st.button('Find Songs', on_click=search_and_go)

# Check for validation page
elif st.session_state.page == 'validation':
    st.title('Select Your Song')

    # Fetch the song info from the session state (tracks_info is set in 'landing' page)
    tracks_info = st.session_state.get('tracks_info', None)

    def go_to_recommendations(song):
        st.session_state.selected_song = song
        st.session_state.page = 'recommendations'

    if tracks_info is not None:
        # Iterate over tracks_info (which is a DataFrame)
        for idx, row in tracks_info.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{row['spotify_title']} - {row['spotify_artist']}")
            with col2:
                # Button to let user choose whether the song matches
                st.button('Use This', 
                        key=f"select_{idx}_{row['spotify_title']}",  # Make the key unique by including the index
                        on_click=go_to_recommendations,
                        args=(row.to_dict(),))  # Pass the row as a dictionary (song details)

        
        # After showing the options, ask if the user has found the song
        user_input = st.text_input('Enter the song or artist name to search again:')
        
        if user_input:
            # Call the search_song function to update tracks_info with new results
            tracks_info = search_song(user_input, (0, 100), (1900, 2024), hot_songs_only=False)
            if tracks_info is not None:
                st.session_state.tracks_info = tracks_info  # Store new results in session state
                st.session_state.page = 'validation'  # Stay on validation page
            else:
                st.write("No songs found. Try searching with different terms.")
                
    # Button to move to recommendations page
    st.button('Next', on_click=lambda: st.session_state.update({'page': 'recommendations'}))

# Check for recommendations page
# Recommendations Page (Page 3)
elif st.session_state.page == 'recommendations':
    st.title('Song Recommendations')

    # Get the selected song from session state
    selected_song = st.session_state.selected_song
    if selected_song:
        st.write(f"You selected: {selected_song['spotify_title']} by {selected_song['spotify_artist']}")

        # Move popularity filter (0-100)
        popularity = st.slider('Select popularity', 0, 100, (0, 100), step=1)

        # Year filter (1900 - 2024)
        year = st.slider('Select year range', 1900, 2024, (1900, 2024), step=1)

        # Checkbox for hot songs only
        hot_songs_only = st.checkbox('Use hot songs only!')

        # Function to refresh recommendations based on filters
        def refresh_recommendations():
            # Call get_recommendations with the selected song and filters
            recommendations = get_recommendations(selected_song, n_recommendations=5, popularity=popularity, year=year, hot_songs_only=hot_songs_only)
            st.session_state.current_recommendations = recommendations  # Store recommendations in session state
            st.experimental_rerun()  # Refresh the app to show new recommendations

        # Button to apply filters and refresh recommendations
        st.button('Apply', on_click=refresh_recommendations)

        # Call get_recommendations to get the recommended songs
        recommendations = st.session_state.get('current_recommendations', get_recommendations(selected_song, n_recommendations=5))

        # Display recommendations
        for idx, row in recommendations.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                # Combine song title and artist in a single line
                st.write(f"**{row['spotify_title']}** - *{row['spotify_artist']}*")
            with col2:
                st.image(row['album_cover'], use_container_width=True)

    # Button to go back to the landing page
    st.button('Back to Search', on_click=lambda: st.session_state.update({'page': 'landing'}))

