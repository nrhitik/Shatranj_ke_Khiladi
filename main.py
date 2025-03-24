import json
import datetime
import google.generativeai as genai  

genai.configure(api_key="AIzaSyA5cpRmMvzqH6avojZ_l0858xyyxdbSvMY")

def generate_travel_prompt(place_name, num_days, budget):
    prompt = f"""
I am planning a {num_days}-day trip to {place_name} with budget {budget}. 

Please provide me with:
* Generate Travel Plan for Location: {place_name}, for {num_days} Days for Couple with a {budget} budget, Give me a Hotels options list with HotelName, Hotel address, Price, hotel image url, geo coordinates, rating, descriptions and suggest itinerary with placeName, Place Details, Place Image Url, Geo Coordinates, ticket Pricing, Time t travel each of the location for 3 days with each day plan with best time to visit in JSON format.

"""
# * A detailed itinerary suggestion, including daily activities and recommended attractions with budget of {budget}.
# * Recommendations for local cuisine and dining experiences in {place_name}.
# * Information on transportation options within {place_name}.
# * Tips for accommodation, including suggested neighborhoods or hotels in {place_name}.

    return prompt

def save_prompt_and_response_to_json(place_name, num_days, budget, prompt, response_text):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{place_name.replace(' ', '_')}_travel_data.json" 
    try:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = [] 

        data.append({
            "place": place_name,
            "days": num_days,
            "budget": budget,
            "prompt": prompt,
            "response": response_text,
            "timestamp": timestamp,
        })

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Prompt and response appended to {filename}")
    except Exception as e:
        print(f"Error saving prompt and response to JSON: {e}")

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')   #### model specify
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting response from Gemini: {e}"

if __name__ == "__main__":
    place = input("Enter the name of the place: ")
    while True:
        try:
            days = int(input("Enter the number of days: "))
            break
        except ValueError:
            print("enter number.")
    while True:
        try:
            budget = input("Enter your budget (cheat or moderate or expensive): ")
            break
        except ValueError:
            print("enter .")

    generated_prompt = generate_travel_prompt(place, days, budget)
    print("\nGenerated Travel Prompt:\n")
    print(generated_prompt)

    gemini_response = get_gemini_response(generated_prompt)
    print("\nGemini Response:\n")
    print(gemini_response)

    save_prompt_and_response_to_json(place, days, budget, generated_prompt, gemini_response)




