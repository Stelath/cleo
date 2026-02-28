import { WiCloud, WiCloudy, WiDayCloudy, WiDaySunny, WiFog, WiRain, WiSnow, WiThunderstorm } from 'react-icons/wi';
import type { HudCard } from '../lib/hud/cards';

type WeatherCardContentProps = {
  card: HudCard;
};

function getMetaValue(card: HudCard, label: string): string {
  const row = card.meta.find((m) => m.label === label);
  return row?.value ?? '';
}

type ConditionCategory = 'clear' | 'partly-cloudy' | 'cloudy' | 'rain' | 'snow' | 'thunderstorm' | 'fog' | 'default';

function categorizeCondition(condition: string): ConditionCategory {
  const lower = condition.toLowerCase();
  if (/thunder|storm/.test(lower)) return 'thunderstorm';
  if (/snow|blizzard|sleet|ice/.test(lower)) return 'snow';
  if (/rain|drizzle|shower/.test(lower)) return 'rain';
  if (/fog|mist|haze/.test(lower)) return 'fog';
  if (/partly cloudy|partly sunny/.test(lower)) return 'partly-cloudy';
  if (/cloud|overcast/.test(lower)) return 'cloudy';
  if (/clear|sunny/.test(lower)) return 'clear';
  return 'default';
}

function getWeatherIcon(category: ConditionCategory) {
  switch (category) {
    case 'clear': return WiDaySunny;
    case 'partly-cloudy': return WiDayCloudy;
    case 'cloudy': return WiCloudy;
    case 'rain': return WiRain;
    case 'snow': return WiSnow;
    case 'thunderstorm': return WiThunderstorm;
    case 'fog': return WiFog;
    default: return WiCloud;
  }
}

function getConditionGradient(category: ConditionCategory): string {
  switch (category) {
    case 'clear': return 'linear-gradient(135deg, #4a90d9 0%, #67b8f0 100%)';
    case 'partly-cloudy': return 'linear-gradient(135deg, #5a9fd4 0%, #8cb8d8 100%)';
    case 'cloudy': return 'linear-gradient(135deg, #7a8a99 0%, #a0adb8 100%)';
    case 'rain': return 'linear-gradient(135deg, #3d5a80 0%, #5c7a99 100%)';
    case 'snow': return 'linear-gradient(135deg, #8faabe 0%, #c8d8e4 100%)';
    case 'thunderstorm': return 'linear-gradient(135deg, #2c3e50 0%, #4a6274 100%)';
    case 'fog': return 'linear-gradient(135deg, #90a4ae 0%, #b0bec5 100%)';
    default: return 'linear-gradient(135deg, #5a9fd4 0%, #8cb8d8 100%)';
  }
}

export default function WeatherCardContent({ card }: WeatherCardContentProps) {
  const temp = getMetaValue(card, 'Temperature');
  const feelsLike = getMetaValue(card, 'Feels Like');
  const humidity = getMetaValue(card, 'Humidity');
  const wind = getMetaValue(card, 'Wind');
  const condition = getMetaValue(card, 'Condition');

  const category = categorizeCondition(condition);
  const Icon = getWeatherIcon(category);
  const gradient = getConditionGradient(category);

  return (
    <article
      className="weather-card"
      data-testid="hud-card-item"
      data-card-id={card.id}
      data-condition={category}
      style={{ '--weather-gradient': gradient } as React.CSSProperties}
    >
      <div className="weather-card-header">
        <Icon className="weather-icon" />
        {temp && <span className="weather-temp-value">{temp}</span>}
      </div>
      {feelsLike && <p className="weather-feels-like">Feels like {feelsLike}</p>}
      {condition && <p className="weather-condition">{condition}</p>}
      {(humidity || wind) && (
        <div className="weather-card-details">
          {humidity && <span>Humidity {humidity}</span>}
          {wind && <span>Wind {wind}</span>}
        </div>
      )}
      {card.title && <p className="weather-location">{card.title.replace(/^Weather\s*-\s*/i, '')}</p>}
    </article>
  );
}
