import React from 'react';

const Profile = () => {
  return (
    <div className='container'>
      <div className='header'>
        <div className='text'>Profile</div>
        <div className='underline'></div>
      </div>
      <div className='profile-details'>
        <h2>Welcome, User!</h2>
        {/* Display user details here */}
      </div>
    </div>
  );
};

export default Profile;
